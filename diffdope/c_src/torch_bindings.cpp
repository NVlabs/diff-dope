/*
 * Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <algorithm>
#include <string>

#define NVDR_CHECK_CUDA_ERROR(CUDA_CALL) { cudaError_t err = CUDA_CALL; AT_CUDA_CHECK(cudaGetLastError()); }
#define NVDR_CHECK_GL_ERROR(GL_CALL) { GL_CALL; GLenum err = glGetError(); TORCH_CHECK(err == GL_NO_ERROR, "OpenGL error: ", getGLErrorString(err), "[", #GL_CALL, ";]"); }
#define CHECK_TENSOR(X, DIMS, CHANNELS) \
    TORCH_CHECK(X.is_cuda(), #X " must be a cuda tensor") \
    TORCH_CHECK(X.scalar_type() == torch::kFloat || X.scalar_type() == torch::kBFloat16, #X " must be fp32 or bf16") \
    TORCH_CHECK(X.dim() == DIMS, #X " must have " #DIMS " dimensions") \
    TORCH_CHECK(X.size(DIMS - 1) == CHANNELS, #X " must have " #CHANNELS " channels")

#include "common.h"
#include "mesh.h"

#define BLOCK_X 8
#define BLOCK_Y 8

//------------------------------------------------------------------------
// mesh.cu

void xfmPointsFwdKernel(XfmKernelParams p);
void xfmPointsBwdKernel(XfmKernelParams p);
void xfmPointsBwdFullKernel(XfmKernelParams p);
void xfmPointsBwdMtxKernel(XfmKernelParams p);
//------------------------------------------------------------------------


//------------------------------------------------------------------------
// Tensor helpers

void update_grid(dim3 &gridSize, torch::Tensor x)
{
    gridSize.x = std::max(gridSize.x, (uint32_t)x.size(2));
    gridSize.y = std::max(gridSize.y, (uint32_t)x.size(1));
    gridSize.z = std::max(gridSize.z, (uint32_t)x.size(0));
}

template<typename... Ts>
void update_grid(dim3& gridSize, torch::Tensor x, Ts&&... vs)
{
    gridSize.x = std::max(gridSize.x, (uint32_t)x.size(2));
    gridSize.y = std::max(gridSize.y, (uint32_t)x.size(1));
    gridSize.z = std::max(gridSize.z, (uint32_t)x.size(0));
    update_grid(gridSize, std::forward<Ts>(vs)...);
}

Tensor make_cuda_tensor(torch::Tensor val)
{
    Tensor res;
    for (int i = 0; i < val.dim(); ++i)
    {
        res.dims[i] = val.size(i);
        res.strides[i] = val.stride(i);
    }
    res.fp16 = val.scalar_type() == torch::kBFloat16;
    res.val = res.fp16 ? (void*)val.data_ptr<torch::BFloat16>() : (void*)val.data_ptr<float>();
    res.d_val = nullptr;
    return res;
}

Tensor make_cuda_tensor(torch::Tensor val, dim3 outDims, torch::Tensor* grad = nullptr)
{
    Tensor res;
    for (int i = 0; i < val.dim(); ++i)
    {
        res.dims[i] = val.size(i);
        res.strides[i] = val.stride(i);
    }
    if (val.dim() == 4)
        res._dims[0] = outDims.z, res._dims[1] = outDims.y, res._dims[2] = outDims.x, res._dims[3] = val.size(3);
    else
        res._dims[0] = outDims.z, res._dims[1] = outDims.x, res._dims[2] = val.size(2), res._dims[3] = 1; // Add a trailing one for indexing math to work out

    res.fp16 = val.scalar_type() == torch::kBFloat16;
    res.val = res.fp16 ? (void*)val.data_ptr<torch::BFloat16>() : (void*)val.data_ptr<float>();
    res.d_val = nullptr;
    if (grad != nullptr)
    {
        if (val.dim() == 4)
            *grad = torch::empty({ outDims.z, outDims.y, outDims.x, val.size(3) }, torch::TensorOptions().dtype(res.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA));
        else // 3
            *grad = torch::empty({ outDims.z, outDims.x, val.size(2) }, torch::TensorOptions().dtype(res.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA));

        res.d_val = res.fp16 ? (void*)grad->data_ptr<torch::BFloat16>() : (void*)grad->data_ptr<float>();
    }
    return res;
}

Tensor make_cuda_tensor_clear(torch::Tensor val, dim3 outDims, unsigned int pad_factor, torch::Tensor* grad = nullptr)
{
    Tensor res;
    for (int i = 0; i < val.dim(); ++i)
    {
        res.dims[i] = val.size(i);
        res.strides[i] = val.stride(i);
    }
    if (val.dim() == 4)
        res._dims[0] = outDims.z, res._dims[1] = outDims.y, res._dims[2] = outDims.x, res._dims[3] = val.size(3);
    else
        res._dims[0] = outDims.z, res._dims[1] = outDims.x, res._dims[2] = val.size(2), res._dims[3] = pad_factor; // Add a trailing one for indexing math to work out

    res.fp16 = val.scalar_type() == torch::kBFloat16;
    res.val = res.fp16 ? (void*)val.data_ptr<torch::BFloat16>() : (void*)val.data_ptr<float>();
    res.d_val = nullptr;
    if (grad != nullptr)
    {
        if (val.dim() == 4)
            *grad = torch::zeros({ outDims.z, outDims.y, outDims.x, val.size(3) }, torch::TensorOptions().dtype(res.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA));
        else // 3
            *grad = torch::zeros({ outDims.z, outDims.x, val.size(2), pad_factor }, torch::TensorOptions().dtype(res.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA));

        res.d_val = res.fp16 ? (void*)grad->data_ptr<torch::BFloat16>() : (void*)grad->data_ptr<float>();
    }
    return res;
}


//------------------------------------------------------------------------
// transform function

torch::Tensor xfm_fwd(torch::Tensor points, torch::Tensor matrix, bool isPoints, bool fp16)
{
    CHECK_TENSOR(points, 3, 3);
    CHECK_TENSOR(matrix, 3, 4);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    XfmKernelParams p;
    p.out.fp16 = fp16;
    p.isPoints = isPoints;
    p.gridSize.x = points.size(1);
    p.gridSize.y = 1;
    p.gridSize.z = std::max(matrix.size(0), points.size(0));

    // Choose launch parameters.
    dim3 blockSize(BLOCK_X * BLOCK_Y, 1, 1);
    dim3 warpSize = getWarpSize(blockSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(p.out.fp16 ? torch::kBFloat16 : torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = isPoints ? torch::empty({ matrix.size(0), points.size(1), 4 }, opts) : torch::empty({ matrix.size(0), points.size(1), 3 }, opts);

    p.points = make_cuda_tensor(points, p.gridSize);
    p.matrix = make_cuda_tensor(matrix, p.gridSize);
    p.out = make_cuda_tensor(out, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)xfmPointsFwdKernel, gridSize, blockSize, args, 0, stream));

    return out;
}

torch::Tensor xfm_bwd(torch::Tensor points, torch::Tensor matrix, torch::Tensor grad, bool isPoints)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    XfmKernelParams p;
    p.isPoints = isPoints;
    p.gridSize.x = points.size(1);
    p.gridSize.y = 1;
    p.gridSize.z = std::max(matrix.size(0), points.size(0));

    // Choose launch parameters.
    dim3 blockSize(BLOCK_X * BLOCK_Y, 1, 1);
    dim3 warpSize = getWarpSize(blockSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    torch::Tensor points_grad;
    p.points = make_cuda_tensor(points, p.gridSize, &points_grad);
    p.matrix = make_cuda_tensor(matrix, p.gridSize);
    p.out = make_cuda_tensor(grad, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)xfmPointsBwdKernel, gridSize, blockSize, args, 0, stream));

    return points_grad;
}

std::tuple<torch::Tensor, torch::Tensor> xfm_bwd_full(torch::Tensor points, torch::Tensor matrix, torch::Tensor grad, bool isPoints)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    XfmKernelParams p;
    p.isPoints = isPoints;
    p.gridSize.x = points.size(1);
    p.gridSize.y = 1;
    p.gridSize.z = std::max(matrix.size(0), points.size(0));

    // Choose launch parameters.
    dim3 blockSize(BLOCK_X * BLOCK_Y, 1, 1);
    dim3 warpSize = getWarpSize(blockSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    dim3 matSize(matrix.size(2), matrix.size(1), matrix.size(0));

    unsigned int padFactor = std::max(std::min(points.size(1), (int64_t)16), std::min( (int64_t)16384, (int64_t)(points.size(1) / 256)));

    p.padFactor = padFactor;
    torch::Tensor points_grad, matrix_grad_pad;
    p.points = make_cuda_tensor(points, p.gridSize, &points_grad);
    p.matrix = make_cuda_tensor_clear(matrix, matSize, padFactor, &matrix_grad_pad);
    p.out = make_cuda_tensor(grad, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)xfmPointsBwdFullKernel, gridSize, blockSize, args, 0, stream));

    // To reduce contention
    torch::Tensor matrix_grad = torch::sum(matrix_grad_pad, 3);

    return std::tuple<torch::Tensor, torch::Tensor>(points_grad, matrix_grad);
}

// Variant that only computes the gradients to matrix elements
torch::Tensor xfm_bwd_mtx(torch::Tensor points, torch::Tensor matrix, torch::Tensor grad, bool isPoints)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Extract input parameters.
    XfmKernelParams p;
    p.isPoints = isPoints;
    p.gridSize.x = points.size(1);
    p.gridSize.y = 1;
    p.gridSize.z = std::max(matrix.size(0), points.size(0));

    // Choose launch parameters.
    dim3 blockSize(BLOCK_X * BLOCK_Y, 1, 1);
    dim3 warpSize = getWarpSize(blockSize);
    dim3 gridSize = getLaunchGridSize(blockSize, p.gridSize);

    dim3 matSize(matrix.size(2), matrix.size(1), matrix.size(0));

    // To reduce contention
    unsigned int padFactor = std::max(std::min(points.size(1), (int64_t)16), std::min( (int64_t)16384, (int64_t)(points.size(1) / 256)));
    p.padFactor = padFactor;

    torch::Tensor points_grad, matrix_grad_pad;
    p.points = make_cuda_tensor(points, p.gridSize);
    p.matrix = make_cuda_tensor_clear(matrix, matSize, padFactor, &matrix_grad_pad);
    p.out = make_cuda_tensor(grad, p.gridSize);

    // Launch CUDA kernel.
    void* args[] = { &p };
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel((const void*)xfmPointsBwdMtxKernel, gridSize, blockSize, args, 0, stream));

    // To reduce contention
    torch::Tensor matrix_grad = torch::sum(matrix_grad_pad, 3);

    return matrix_grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xfm_fwd", &xfm_fwd, "xfm_fwd");
    m.def("xfm_bwd", &xfm_bwd, "xfm_bwd");
    m.def("xfm_bwd_full", &xfm_bwd_full, "xfm_bwd_full");
    m.def("xfm_bwd_mtx", &xfm_bwd_mtx, "xfm_bwd_mtx");
}
