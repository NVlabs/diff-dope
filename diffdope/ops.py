# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import sys

import numpy as np
import torch
import torch.utils.cpp_extension

# ----------------------------------------------------------------------------
# C++/Cuda plugin compiler/loader.

_cached_plugin = None


def _get_plugin():
    # Return cached plugin if already loaded.
    global _cached_plugin
    if _cached_plugin is not None:
        return _cached_plugin

    # Make sure we can find the necessary compiler and library binaries.
    if os.name == "nt":

        def find_cl_path():
            import glob

            for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
                paths = sorted(
                    glob.glob(
                        r"C:\Program Files (x86)\Microsoft Visual Studio\*\%s\VC\Tools\MSVC\*\bin\Hostx64\x64"
                        % edition
                    ),
                    reverse=True,
                )
                if paths:
                    return paths[0]

        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError(
                    "Could not locate a supported Microsoft Visual C++ installation"
                )
            os.environ["PATH"] += ";" + cl_path

    # Compiler options.
    opts = ["-DNVDR_TORCH"]

    # Linker options.
    if os.name == "posix":
        ldflags = ["-lcuda", "-lnvrtc"]
    elif os.name == "nt":
        ldflags = ["cuda.lib", "advapi32.lib", "nvrtc.lib"]

    # List of sources.
    source_files = ["c_src/mesh.cu", "c_src/common.cpp", "c_src/torch_bindings.cpp"]

    # Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
    os.environ["TORCH_CUDA_ARCH_LIST"] = ""

    # Try to detect if a stray lock file is left in cache directory and show a warning. This sometimes happens on Windows if the build is interrupted at just the right moment.
    try:
        lock_fn = os.path.join(
            torch.utils.cpp_extension._get_build_directory("renderutils_plugin", False),
            "lock",
        )
        if os.path.exists(lock_fn):
            print("Warning: Lock file exists in build directory: '%s'" % lock_fn)
    except:
        pass

    # Compile and load.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    torch.utils.cpp_extension.load(
        name="renderutils_plugin",
        sources=source_paths,
        extra_cflags=opts,
        extra_cuda_cflags=opts,
        extra_ldflags=ldflags,
        with_cuda=True,
        verbose=True,
    )

    # Import, cache, and return the compiled module.
    import renderutils_plugin

    _cached_plugin = renderutils_plugin
    return _cached_plugin


# ----------------------------------------------------------------------------
# Transform points function


class _xfm_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, matrix, isPoints):
        ctx.save_for_backward(points, matrix)
        ctx.isPoints = isPoints
        return _get_plugin().xfm_fwd(points, matrix, isPoints, False)

    @staticmethod
    def backward(ctx, dout):
        points, matrix = ctx.saved_variables
        matrix_grad = None
        points_grad = None
        if matrix.requires_grad and points.requires_grad:
            points_grad, matrix_grad = _get_plugin().xfm_bwd_full(
                points, matrix, dout, ctx.isPoints
            )
        elif matrix.requires_grad and not points.requires_grad:
            matrix_grad = _get_plugin().xfm_bwd_mtx(points, matrix, dout, ctx.isPoints)
        else:
            points_grad = _get_plugin().xfm_bwd(points, matrix, dout, ctx.isPoints)

        return points_grad, matrix_grad, None, None


def xfm_points(points, matrix, use_python=False):
    """Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    """
    if use_python:
        out = torch.matmul(
            torch.nn.functional.pad(points, pad=(0, 1), mode="constant", value=1.0),
            torch.transpose(matrix, 1, 2),
        )
    else:
        out = _xfm_func.apply(points, matrix, True)

    if torch.is_anomaly_enabled():
        assert torch.all(
            torch.isfinite(out)
        ), "Output of xfm_points contains inf or NaN"
    return out


def xfm_vectors(vectors, matrix, use_python=False):
    """Transform vectors.
    Args:
        vectors: Tensor containing 3D vectors with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)

    Returns:
        Transformed vectors in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    """

    if use_python:
        out = torch.matmul(
            torch.nn.functional.pad(vectors, pad=(0, 1), mode="constant", value=0.0),
            torch.transpose(matrix, 1, 2),
        )[..., 0:3].contiguous()
    else:
        out = _xfm_func.apply(vectors, matrix, False)

    if torch.is_anomaly_enabled():
        assert torch.all(
            torch.isfinite(out)
        ), "Output of xfm_vectors contains inf or NaN"
    return out
