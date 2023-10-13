# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch

import os
import sys
sys.path.insert(0, os.path.join(sys.path[0], '../..'))
import renderutils as ru

DTYPE=torch.float32

def test_xfm(BATCH, RES, ITR):
    print("------Testing xfm_point --------")
    points_cuda = torch.rand(1, RES, 3, dtype=DTYPE, device='cuda', requires_grad=True)
    points_ref = points_cuda.clone().detach().requires_grad_(True)
    mtx_cuda = torch.rand(BATCH, 4, 4, dtype=DTYPE, device='cuda', requires_grad=True)
    mtx_ref = mtx_cuda.clone().detach().requires_grad_(True)
    target = torch.rand(BATCH, RES, 4, dtype=DTYPE, device='cuda', requires_grad=True)

    ref_out = ru.xfm_points(points_ref, mtx_ref, use_python=True)
    cuda_out = ru.xfm_points(points_cuda, mtx_cuda)

    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)

    print("--- Testing: [%d, %d] ---" % (BATCH, RES))

    # Torch
    a.record()
    for i in range(ITR):
        ref_out = ru.xfm_points(points_ref, mtx_ref, use_python=True)
    b.record()
    torch.cuda.synchronize()
    print("FWD     xfm Python:", a.elapsed_time(b)/ITR)

    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)

    a.record()
    for i in range(ITR):
        ref_out = ru.xfm_points(points_ref, mtx_ref, use_python=True)
        ref_out.backward(torch.ones_like(ref_out)*0.12345*i)
    
    b.record()
    torch.cuda.synchronize()
    print("FWD+BWD xfm Python:", a.elapsed_time(b)/ITR)

    # Cuda
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)

    a.record()
    for i in range(ITR):
        cuda_out = ru.xfm_points(points_cuda, mtx_cuda)

    b.record()
    torch.cuda.synchronize()
    print("FWD     xfm Cuda  :", a.elapsed_time(b)/ITR)

    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for i in range(ITR):
        cuda_out = ru.xfm_points(points_cuda, mtx_cuda)
        cuda_out.backward(torch.ones_like(cuda_out)*0.12345*i)

    b.record()
    torch.cuda.synchronize()
    print("FWD+BWD xfm Cuda  :", a.elapsed_time(b)/ITR)


test_xfm(8, 512*512, 1000)
#test_loss(16, 512, 1000)
#test_loss(1, 2048, 1000)
