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
import renderutils_plugin

# ----------------------------------------------------------------------------
# Transform points function


class _xfm_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, matrix, isPoints):
        ctx.save_for_backward(points, matrix)
        ctx.isPoints = isPoints
        return renderutils_plugin.xfm_fwd(points, matrix, isPoints, False)

    @staticmethod
    def backward(ctx, dout):
        points, matrix = ctx.saved_variables
        matrix_grad = None
        points_grad = None
        if matrix.requires_grad and points.requires_grad:
            points_grad, matrix_grad = renderutils_plugin.xfm_bwd_full(
                points, matrix, dout, ctx.isPoints
            )
        elif matrix.requires_grad and not points.requires_grad:
            matrix_grad = renderutils_plugin.xfm_bwd_mtx(points, matrix, dout, ctx.isPoints)
        else:
            points_grad = renderutils_plugin.xfm_bwd(points, matrix, dout, ctx.isPoints)

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
