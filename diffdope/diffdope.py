import logging

import matplotlib

matplotlib.use("Agg")

import collections
import io
import sys
import math
import pathlib
import random
import warnings
from dataclasses import dataclass
from itertools import repeat
from types import FunctionType
from typing import Any, BinaryIO, List, Optional, Tuple, Union

import cv2
import hydra
import hydra.utils as hydra_utils
import imageio
import matplotlib.pyplot as plt
import numpy as np
import nvdiffrast.torch as dr
import pyrr
import torch
import trimesh
from icecream import ic
from omegaconf import DictConfig, OmegaConf
from PIL import Image as pilImage
from PIL import ImageColor, ImageDraw, ImageFont
from tqdm import tqdm

import diffdope as dd

# for better print debug
print()
if not hasattr(sys, 'ps1'):
    print = ic

# A logger for this file
log = logging.getLogger(__name__)


def matrix_batch_44_from_position_quat(q, p):
    """
    Convert a batched position and quaternion into a batch matrix while keeping the gradients intact.

    Args:
        p (torch.tensor): (batch,3) translation vector
        q (torch.tensor): (batch,4) quaternion in x,y,z,w

    Return:
        returns a (batch,4,4) torch tensor that keeps the gradients intact
    """
    r0 = torch.stack(
        [
            1.0 - 2.0 * q[:, 1] ** 2 - 2.0 * q[:, 2] ** 2,
            2.0 * q[:, 0] * q[:, 1] - 2.0 * q[:, 2] * q[:, 3],
            2.0 * q[:, 0] * q[:, 2] + 2.0 * q[:, 1] * q[:, 3],
        ],
        dim=1,
    )
    r1 = torch.stack(
        [
            2.0 * q[:, 0] * q[:, 1] + 2.0 * q[:, 2] * q[:, 3],
            1.0 - 2.0 * q[:, 0] ** 2 - 2.0 * q[:, 2] ** 2,
            2.0 * q[:, 1] * q[:, 2] - 2.0 * q[:, 0] * q[:, 3],
        ],
        dim=1,
    )
    r2 = torch.stack(
        [
            2.0 * q[:, 0] * q[:, 2] - 2.0 * q[:, 1] * q[:, 3],
            2.0 * q[:, 1] * q[:, 2] + 2.0 * q[:, 0] * q[:, 3],
            1.0 - 2.0 * q[:, 0] ** 2 - 2.0 * q[:, 1] ** 2,
        ],
        dim=1,
    )
    rr = torch.stack([r0, r1, r2], dim=1)
    aa = torch.stack([p[:, 0], p[:, 1], p[:, 2]], dim=1).reshape(-1, 3, 1)
    rr = torch.cat([rr, aa], dim=2)  # Pad right column.
    aa = torch.stack(
        [torch.tensor([0, 0, 0, 1], dtype=torch.float32).cuda()] * aa.shape[0], dim=0
    )
    rr = torch.cat([rr, aa.reshape(-1, 1, 4)], dim=1)  # Pad bottom row.

    return rr


def opencv_2_opengl(p, q):
    """
    Converts a pose from the OpenCV coordinate system to the OpenGL coordinate system.

    Args:
        p (np.ndarray): position
        q (pyrr.Quaternion): quat

    Returns:
        p,q
    """
    source_transform = q.matrix44
    source_transform[:3, 3] = p
    opengl_to_opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )

    R_opengl_to_opencv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    t_opengl_to_opencv = np.array([0, 0, 0])

    # Adjust rotation and translation for target coordinate system
    adjusted_rotation = np.dot(R_opengl_to_opencv, source_transform[:3, :3])
    adjusted_translation = (
        np.dot(R_opengl_to_opencv, source_transform[:3, 3]) + t_opengl_to_opencv
    )

    # Build target transformation matrix (OpenCV convention)
    target_transform = np.eye(4)
    target_transform[:3, :3] = adjusted_rotation
    target_transform[:3, 3] = adjusted_translation

    q = pyrr.Matrix44(target_transform).quaternion

    # TODO verify what is going on here, this should not be needed ...
    # legacy code here
    q = (
        q
        * pyrr.Quaternion.from_z_rotation(np.pi / 2)
        * pyrr.Quaternion.from_y_rotation(-np.pi / 2)
    )
    q = (
        q
        * pyrr.Quaternion.from_z_rotation(-np.pi / 2)
        * pyrr.Quaternion.from_x_rotation(-np.pi / 2)
    )
    # END TODO

    return target_transform[:3, 3], q


def interpolate(attr, rast, attr_idx, rast_db=None):
    """
    A wrapper around nvdiffrast interpolate
    """
    return dr.interpolate(
        attr.contiguous(),
        rast,
        attr_idx,
        rast_db=rast_db,
        diff_attrs=None if rast_db is None else "all",
    )


def render_texture_batch(
    glctx,
    proj_cam,
    mtx,
    pos,
    pos_idx,
    resolution,
    uv=None,
    uv_idx=None,
    tex=None,
    vtx_color=None,
    return_rast_out=False,
):
    """
    Functions that maps 3d objects to the nvdiffrast rendering. This function uses the color of the texture of the model or the vertex color.
    If there is no illumination on the object model then it will look flat, we recommend to bake in some lights, see blender script provided.
    See the mesh class if you want more information about how to construct your mesh for rendering.

    Args:
    glctx (): the nvdiffrast context
    proj_cam (torch.tensor): (b,4,4) is the camera projection matrix
    mtx (torch.tensor): (b,4,4) the world camera pose express in opengl coordinate system
    pos (torch.tensor): (b,nb_points,3) the object 3d points defining the 3d model
    pos_idx (torch.tensor): (nb_points,3) the object triangle list
    resolution (np.ndarray): (2) the image resolution to be render
    uv (torch.tensor): (b,nb_points,2) where each object point lands on the texture
    uv_idx (torch.tensor): (b,nb_points,3) defining each texture triangle
    tex (torch.tensor): (b,w,h,3) batch image of the texture
    vtx_color (torch.tensor): (b,nb_points,3) the color of each vertex, this is used when the texture is not defined
    return_rast_out (bool): return the nvdiffrast output

    Return:
        returns a dict with key 'rgb','depth', and 'rast_out'
    """
    if not type(resolution) == list:
        resolution = [resolution, resolution]
    posw = torch.cat([pos, torch.ones([pos.shape[0], pos.shape[1], 1]).cuda()], axis=2)
    mtx_transpose = torch.transpose(mtx, 1, 2)

    final_mtx_proj = torch.matmul(proj_cam, mtx)
    pos_clip_ja = dd.xfm_points(pos.contiguous(), final_mtx_proj)

    rast_out, rast_out_db = dr.rasterize(
        glctx, pos_clip_ja, pos_idx[0], resolution=resolution
    )

    # compute the depth
    gb_pos, _ = interpolate(posw, rast_out, pos_idx[0], rast_db=rast_out_db)
    shape_keep = gb_pos.shape
    gb_pos = gb_pos.reshape(shape_keep[0], -1, shape_keep[-1])
    gb_pos = gb_pos[..., :3]

    depth = dd.xfm_points(gb_pos.contiguous(), mtx)
    depth = depth.reshape(shape_keep)[..., 2] * -1

    # mask   , _ = dr.interpolate(torch.ones(pos_idx.shape).cuda(), rast_out, pos_idx)
    mask, _ = dr.interpolate(torch.ones(pos_idx.shape).cuda(), 
        rast_out, pos_idx[0],rast_db=rast_out_db,diff_attrs="all")
    mask       = dr.antialias(mask, rast_out, pos_clip_ja, pos_idx[0])

    # compute vertex color interpolation
    if vtx_color is None:
        texc, texd = dr.interpolate(
            uv, rast_out, uv_idx[0], rast_db=rast_out_db, diff_attrs="all"
        )
        color = dr.texture(
            tex,
            texc,
            texd,
            filter_mode="linear",
        )

        color = color * torch.clamp(rast_out[..., -1:], 0, 1)  # Mask out background.
    else:
        color, _ = dr.interpolate(vtx_color, rast_out, pos_idx[0])
        color = color * torch.clamp(rast_out[..., -1:], 0, 1)  # Mask out background.
    if not return_rast_out:
        rast_out = None
    return {"rgb": color, "depth": depth, "rast_out": rast_out, 'mask':mask}


##############################################################################
# IMG MANIPULATION
##############################################################################


@torch.no_grad()
def find_crop(img_tensor, percentage=0.1):
    """
    Find the bounding crop in an image, assuming it finds where there is non-zero.

    Args:
        img_tensor: Input image tensor

    Returns:
        list: [top_row, left_col, size]
    """

    img_tensor = (img_tensor > 0).float()

    # Find the bounding box of the img_tensor
    rows, cols = torch.nonzero(img_tensor[..., 0], as_tuple=True)
    top_row, left_col = rows.min(), cols.min()
    bottom_row, right_col = rows.max(), cols.max()

    # Calculate the wiggle room for each dimension (percentage of the width/height)
    wiggle_room_rows = int((bottom_row - top_row + 1) * percentage)
    wiggle_room_cols = int((right_col - left_col + 1) * percentage)

    # Expand the bounding box by the wiggle room
    top_row = max(0, top_row - wiggle_room_rows)
    left_col = max(0, left_col - wiggle_room_cols)
    bottom_row = min(img_tensor.shape[0] - 1, bottom_row + wiggle_room_rows)
    right_col = min(img_tensor.shape[1] - 1, right_col + wiggle_room_cols)

    # Calculate the size of the square crop as the minimum of the expanded height and width
    crop_size = max(bottom_row - top_row, right_col - left_col)

    return [top_row, left_col, crop_size]


def getimg_stack(color_imgs, depth=False, depth_max=3, w=1, h=1):
    if depth:
        for i_im in range(len(color_imgs)):
            color_imgs[i_im] = torch.cat(
                [
                    color_imgs[i_im].unsqueeze(-1),
                    color_imgs[i_im].unsqueeze(-1),
                    color_imgs[i_im].unsqueeze(-1),
                ],
                dim=-1,
            )

            color_imgs[i_im][color_imgs[i_im] < 0] = depth_max

            color_imgs[i_im] /= depth_max
            # print(color_imgs[i_im].shape)

    col_imgs = []
    for ii in range(h):
        row_imgs = []
        for jj in range(w):
            if ii + jj < len(color_imgs):
                img_ref = color_imgs[ii + jj][0].detach().cpu().numpy()
            else:
                img_ref = np.zeros(color_imgs[-1][0].shape)
            row_imgs.append(img_ref)

        row_all = np.concatenate(row_imgs, axis=1)[::-1]
        # print(row_all.shape)
        col_imgs.append(row_all)
    gt_final = np.concatenate(col_imgs, axis=0)
    # return cv2.resize(gt_final,(400,400))
    return gt_final


@torch.no_grad()
def im_resize(image, width=None, height=None):
    dim = None
    (h, w) = image.shape[:2]

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    elif height is None:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim)
    return resized


@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Make a grid of images.
    taken from https://github.com/pytorch/vision/blob/main/torchvision/utils.py

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Returns:
        grid (Tensor): the tensor containing grid of images.
    """
    if not torch.is_tensor(tensor):
        if isinstance(tensor, list):
            for t in tensor:
                if not torch.is_tensor(t):
                    raise TypeError(
                        f"tensor or list of tensors expected, got a list containing {type(t)}"
                    )
        else:
            raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError(
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"
            )

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor should be of type torch.Tensor")
    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full(
        (num_channels, height * ymaps + padding, width * xmaps + padding), pad_value
    )
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(
                tensor[k]
            )
            k = k + 1
    return grid


@torch.no_grad()
def make_grid_image(img_batch, row, final_width, depth=False):
    img_batch = make_grid(img_batch.permute(0, 3, 1, 2), nrow=row)
    img_batch = (
        img_batch.mul(255)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    img_batch = cv2.cvtColor(img_batch, cv2.COLOR_BGR2RGB)
    if depth is True:
        img_batch = cv2.applyColorMap((img_batch).astype(np.uint8), cv2.COLORMAP_JET)
    img_batch = im_resize(img_batch, width=final_width)

    return img_batch


@torch.no_grad()
def make_grid_overlay_batch(
    foreground,
    background=None,
    alpha=0.5,
    row=2,
    final_width=2000,
    add_background=True,
    add_contour=True,
    color_countour=[1, 0, 0],
    flip_result=True,
):
    """
    Make a grid image of a batch

    Args:
        foreground (torch.tensor): BxWxHx3 normalized foreground image
        background (torch.tensor): BxWxHx3 normalized background image, if None then don't add
        alpha (float): alpha of the nvdiffrast render
        row (int): how entry in a row you want for the grid
        final_width (int): final width in pixel of the grid image
        add_background (bool): Add the background or not
        add_contour (bool): Add the foreground contour or not
        color_countour (list(float)): color in normalized space
        flip_result (bool): Flip the final image

    Returns:
        A grid cv2 image (nd.array): WxHx3
    """

    # assumes a 3 channel image and normalized
    foreground = make_grid_image(foreground, row, final_width)

    if add_contour:
        alpha_img = np.zeros(foreground.shape[:2])

        gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.uint8)
        alpha_img[gray > 0] = alpha

        cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if not background is None and add_background:
        background = make_grid_image(background, row, final_width)
    else:
        # make a black image
        background = np.zeros(foreground.shape)

    blended_image = cv2.merge(
        (
            alpha_img * foreground[:, :, 0] + (1 - alpha_img) * background[:, :, 0],
            alpha_img * foreground[:, :, 1] + (1 - alpha_img) * background[:, :, 1],
            alpha_img * foreground[:, :, 2] + (1 - alpha_img) * background[:, :, 2],
        )
    ).astype("uint8")

    if add_contour:
        for c in cnts:
            cv2.drawContours(
                blended_image, [c], -1, (36, 255, 12), thickness=1, lineType=cv2.LINE_AA
            )
    if flip_result:
        blended_image = cv2.flip(blended_image, 0)

    return blended_image


##############################################################################
# LOSSES
##############################################################################
def dist_batch_lr(tensor, learning_rates, channels=[1, 2, 3]):
    """
    Method that distribute different learning rates to the batch.

    Args:
        tensor (torch.tensor): BxWxHxC if you do not have C, pass a different set of channel, for example you could run on a depth map [1,2].
        learning_rates (torch.tensor): B the different learning rates needed.
        channels (list): the index values used to apply the first mean, e.g., [1,2,3] for colored image, or [1,2] for a depth map
    """

    return torch.mean(tensor, channels) * learning_rates


def l1_rgb_with_mask(ddope):
    """
    Computes the l1_rgb on a DiffDOPE object, simpler to pass the object.
    """

    diff_rgb = torch.abs(
        (ddope.renders["rgb"] - ddope.gt_tensors["rgb"])
        * ddope.gt_tensors["segmentation"]
    )
    lr_diff_rgb = dist_batch_lr(diff_rgb, ddope.learning_rates)

    ddope.add_loss_value(
        "rgb", torch.mean(diff_rgb.detach(), (1, 2, 3)) * ddope.cfg.losses.weight_rgb
    )

    return lr_diff_rgb.mean() * ddope.cfg.losses.weight_rgb


def l1_depth_with_mask(ddope):
    """
    Computes the l1_depth on a DiffDOPE object, simpler to pass the object.
    """

    diff_depth = torch.abs(
        (ddope.renders["depth"] - ddope.gt_tensors["depth"])
        * ddope.gt_tensors["segmentation"][..., 0]
    )
    lr_diff_depth = dist_batch_lr(diff_depth, ddope.learning_rates, [1, 2])

    ddope.add_loss_value(
        "depth", torch.mean(diff_depth.detach(), (1, 2)) * ddope.cfg.losses.weight_depth
    )

    return lr_diff_depth.mean() * ddope.cfg.losses.weight_depth


def l1_mask(ddope):
    """
    Computes the L1-on mask on a DiffDOPE object.

    Args:
    - ddope: A DiffDOPE object containing the necessary data.

    Returns:
    - lr_diff_mask_mean: The mean of the L1-on mask loss multiplied by the weight specified in the configuration.
    """

    mask = ddope.renders["mask"]
    ddope.optimization_results[-1]["mask"] = mask.detach().cpu()

    # Compute the difference between the mask and ground truth segmentation
    diff_mask = torch.abs(mask - ddope.gt_tensors["segmentation"])

    # Compute the L1-on mask loss with batch-wise learning rates
    lr_diff_mask = dist_batch_lr(diff_mask, ddope.learning_rates)

    # Add the L1-on mask loss to the DiffDOPE object
    ddope.add_loss_value(
        "mask_selection",
        torch.mean(torch.abs(diff_mask.detach()), (1, 2, 3))
        * ddope.cfg.losses.weight_mask,
    )

    # Calculate the mean of the L1-on mask loss and apply the weight from the configuration
    lr_diff_mask_mean = lr_diff_mask.mean() * ddope.cfg.losses.weight_mask

    return lr_diff_mask_mean


##############################################################################
# CLASSES
##############################################################################


@dataclass
class Camera:
    """
    A class for representing the camera, mostly to store classic computer vision oriented reprojection values
    and then get the OpenGL projection matrix out.

    Args:
        fx (float): focal length x-axis in pixel unit
        fy (float): focal length y-axis in pixel unit
        cx (float): principal point x-axis in pixel
        cy (float): principal point y-axis in pixel
        im_width (int): width of the image
        im_height (int): height of the image
        znear (float, optional): for the opengl reprojection, how close can a point be to the camera before it is clipped
        zfar (float, optional): for the opengl reprojection, how far can a point be to the camera before it is clipped
    """

    fx: float
    fy: float
    cx: float
    cy: float
    im_width: int
    im_height: int
    znear: Optional[float] = 0.01
    zfar: Optional[float] = 200

    def __post_init__(self):
        self.cam_proj = self.get_projection_matrix()

    def set_batchsize(self, batchsize):
        """
        Change the batchsize for the image tensor

        Args:
            batchsize (int): batchsize for the tensor
        """
        if len(self.cam_proj.shape) == 2:
            self.cam_proj = torch.stack([self.cam_proj] * batchsize, dim=0)
        else:
            self.cam_proj = torch.stack([self.cam_proj[0]] * batchsize, dim=0)

    def cuda(self):
        self.cam_proj = self.cam_proj.cuda().float()

    def resize(self, percentage):
        """
        If you resize the images for the optimization

        Args:
            percentage (float): bounded between [0,1]
        """
        self.fx *= percentage
        self.fy *= percentage
        self.cx = (int)(percentage * self.cx)
        self.cy = (int)(percentage * self.cy)
        self.im_width = (int)(percentage * self.im_width)
        self.im_height = (int)(percentage * self.im_height)

    def get_projection_matrix(self):
        """
        Conversion of Hartley-Zisserman intrinsic matrix to OpenGL projection matrix.

        Refs:

        1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
        2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py
        3) https://github.com/megapose6d/megapose6d/blob/3f5b51d9cef71d9ac0ac36c6414f35013bee2b0b/src/megapose/panda3d_renderer/types.py

        Returns:
            torch.tensor: a 4x4 projection matrix in OpenGL coordinate frame
        """

        K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        x0 = 0
        y0 = 0
        w = self.im_width
        h = self.im_height
        nc = self.znear
        fc = self.zfar

        window_coords = "y_down"

        depth = float(fc - nc)
        q = -(fc + nc) / depth
        qn = -2 * (fc * nc) / depth

        # Draw our images upside down, so that all the pixel-based coordinate
        # systems are the same.
        if window_coords == "y_up":
            proj = np.array(
                [
                    [
                        2 * K[0, 0] / w,
                        -2 * K[0, 1] / w,
                        (-2 * K[0, 2] + w + 2 * x0) / w,
                        0,
                    ],
                    [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
                    [0, 0, q, qn],  # Sets near and far planes (glPerspective).
                    [0, 0, -1, 0],
                ]
            )

        # Draw the images upright and modify the projection matrix so that OpenGL
        # will generate window coords that compensate for the flipped image coords.
        else:
            assert window_coords == "y_down"
            proj = np.array(
                [
                    [
                        2 * K[0, 0] / w,
                        -2 * K[0, 1] / w,
                        (-2 * K[0, 2] + w + 2 * x0) / w,
                        0,
                    ],
                    [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
                    [0, 0, q, qn],  # Sets near and far planes (glPerspective).
                    [0, 0, -1, 0],
                ]
            )

        return torch.tensor(proj)


# @dataclass
class Mesh(torch.nn.Module):
    """
    A wrapper around a Trimesh mesh, where the data is already loaded
    to be consumed by PyTorch. As such the internal values are stored as
    torch array.

    Args:
        path_model (str): path to the object to be loaded, see Trimesh which extensions are supported.
        scale (int): scale of mesh

    Attributes:
        pos_idx (torch.tensor): (nb_triangle,3) triangle list for the mesh
        pos (torch.tensor): (n,3) vertex positions in object space
        vtx_color (torch.tensor): (n,3) vertex color - might not exists if the file does not have that information stored
        tex (torch.tensor): (w,h,3) textured saved - might not exists if the file does not have texture
        uv (torch.tensor): (n,2) vertex uv position - might not exists if the file does not have texture
        uv_idx (torch.tensor): (nb_triangle,3) triangles for the uvs - might not exists if the file does not have texture
        bounding_volume (np.array): (2,3) minimum x,y,z with maximum x,y,z
        dimensions (list): size in all three axes
        center_point (list): position of the center of the object
        textured_map (boolean): was there a texture loaded
    """

    def __init__(self, path_model, scale):
        super().__init__()

        # load the mesh
        self.path_model = path_model
        self.to_process = [
            "pos",
            "pos_idx",
            "vtx_color",
            "tex",
            "uv",
            "uv_idx",
            "vtx_normals",
        ]

        mesh = trimesh.load(self.path_model, force="mesh")

        pos = np.asarray(mesh.vertices)
        pos_idx = np.asarray(mesh.faces)

        normals = np.asarray(mesh.vertex_normals)

        pos_idx = torch.from_numpy(pos_idx.astype(np.int32))

        vtx_pos = torch.from_numpy(pos.astype(np.float32)) * scale
        vtx_normals = torch.from_numpy(normals.astype(np.float32))
        bounding_volume = [
            [
                torch.min(vtx_pos[:, 0]),
                torch.min(vtx_pos[:, 1]),
                torch.min(vtx_pos[:, 2]),
            ],
            [
                torch.max(vtx_pos[:, 0]),
                torch.max(vtx_pos[:, 1]),
                torch.max(vtx_pos[:, 2]),
            ],
        ]

        dimensions = [
            bounding_volume[1][0] - bounding_volume[0][0],
            bounding_volume[1][1] - bounding_volume[0][1],
            bounding_volume[1][2] - bounding_volume[0][2],
        ]
        center_point = [
            ((bounding_volume[0][0] + bounding_volume[1][0]) / 2).item(),
            ((bounding_volume[0][1] + bounding_volume[1][1]) / 2).item(),
            ((bounding_volume[0][2] + bounding_volume[1][2]) / 2).item(),
        ]

        if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
            tex = np.array(mesh.visual.material.image) / 255.0
            uv = mesh.visual.uv
            uv[:, 1] = 1 - uv[:, 1]
            uv_idx = np.asarray(mesh.faces)

            tex = torch.from_numpy(tex.astype(np.float32))
            uv_idx = torch.from_numpy(uv_idx.astype(np.int32))
            vtx_uv = torch.from_numpy(uv.astype(np.float32))

            self.pos_idx = pos_idx
            self.pos = vtx_pos
            self.tex = tex
            self.uv = vtx_uv
            self.uv_idx = uv_idx
            self.bounding_volume = bounding_volume
            self.dimensions = dimensions
            self.center_point = center_point
            self.vtx_normals = vtx_normals
            self.has_textured_map = True

        else:
            vertex_color = mesh.visual.vertex_colors[..., :3] / 255.0
            vertex_color = torch.from_numpy(vertex_color.astype(np.float32))

            self.pos_idx = pos_idx
            self.pos = vtx_pos
            self.vtx_color = vertex_color
            self.bounding_volume = bounding_volume
            self.dimensions = dimensions
            self.center_point = center_point
            self.vtx_normals = vtx_normals
            self.has_textured_map = False

        log.info(
            f"loaded mesh @{self.path_model}. Does it have texture map? {self.has_textured_map} "
        )
        self._batchsize_set = False

    def __str__(self):
        return f"mesh @{self.path_model}. vtx:{self.pos.shape} on {self.pos.device}"

    def __repr__(self):
        return f"mesh @{self.path_model}. vtx:{self.pos.shape} on {self.pos.device}"

    def set_batchsize(self, batchsize):
        """
        Set the batchsize of the mesh object to match the optimization.

        Args:
            batchsize (int): batchsize for the arrays used by nv diff rast

        """

        # TODO tex does not seem to be correct in its shape

        for key, value in vars(self).items():
            if not key in self.to_process:
                continue
            if self._batchsize_set is False:
                vars(self)[key] = torch.stack([vars(self)[key]] * batchsize, dim=0)
            else:
                vars(self)[key] = torch.stack([vars(self)[key][0]] * batchsize, dim=0)

        for key, value in self._parameters.items():
            if not key in self.to_process:
                continue
            if self._batchsize_set is False:
                self._parameters[key] = torch.stack(
                    [self._parameters[key]] * batchsize, dim=0
                )
            else:
                self._parameters[key] = torch.stack(
                    [self._parameters[key][0]] * batchsize, dim=0
                )

        if self._batchsize_set is False:
            self._batchsize_set = True

    def cuda(self):
        """
        put the arrays from `to_process` on gpu
        """
        super().cuda()

        for key, value in vars(self).items():
            if not key in self.to_process:
                continue
            vars(self)[key] = vars(self)[key].cuda()

    def enable_gradients_texture(self):
        """
        Function to enable gradients on the texture *please note* if `set_batchsize` is called after this function the gradients are set to false for the image automatically
        """
        if self.has_textured_map:
            self.tex = torch.nn.Parameter(self.tex, requires_grad=True).to(
                self.tex.device
            )
        else:
            self.vtx_color = torch.nn.Parameter(self.vtx_color, requires_grad=True).to(
                self.vtx_color.device
            )

    def forward(self):
        """
        Pass the information from the mesh back to diff-dope defined in the the `to_process`
        """
        to_return = {}
        for key, value in vars(self).items():
            if not key in self.to_process:
                continue
            to_return[key] = vars(self)[key]
        # if not 'tex' in to_return:
        #     to_return['tex'] = self.tex
        # elif not 'vtx_color' in to_return:
        #     to_return['vtx_color'] = self.vtx_color
        return to_return


class Object3D(torch.nn.Module):
    """
    This is the 3D object we want to run optimization on representation that Diff-DOPE uses to optimize.

    Attributes:
        qx,qy,qz,qw (torch.nn.Parameter): batchsize x 1 representing the quaternion
        x,y,z (torch.nn.Parameter): batchsize x 1 representing the position
        mesh (Mesh): batchsize x width x height representing the object texture, can be use in the optimization

    """

    def __init__(
        self,
        position: list,
        rotation: list,
        batchsize: int = 32,
        opencv2opengl: bool = True,
        model_path: str = None,
        scale: int = 1,
    ):
        """
        Args:
            position (list): a 3 value list of the object position
            rotation (list): could be a quat with 4 values (x,y,z,w), or a flatten rotational matrix or a 3x3 matrix (as a list of list) -- both are row-wise / row-major.
            batchsize (int): size of the batch to be optimized, this is defined normally as a hyperparameter.
            opencv2opengl (bool): converting the coordinate space from one to the other
            scale (int): scale to apply to the position
        """
        super().__init__()
        self.qx = None  # to load on cpu and not gpu

        if model_path is None:
            self.mesh = None    
        else:
            self.mesh = Mesh(path_model=model_path, scale=scale)

        self.set_pose(
            position, rotation, batchsize, scale=scale, opencv2opengl=opencv2opengl
        )

    def set_pose(
        self,
        position: list,
        rotation: list,
        batchsize: int = 32,
        opencv2opengl: bool = True,
        scale: int = 1,
    ):
        """
        Set the pose to new values, the inputs can be either list, numpy or torch.tensor. If the class was put on cuda(), the updated pose should be on the GPU as well.

        Args:
            position (list): a 3 value list of the object position
            rotation (list): could be a quat with 4 values (x,y,z,w), or a flatten rotational matrix or a 3x3 matrix (as a list of list) -- both are row-wise / row-major.
            batchsize (int): size of the batch to be optimized, this is defined normally as a hyperparameter.
            scale (int): scale to apply to the position

        """

        assert len(position) == 3
        position = np.array(position) * scale

        assert len(rotation) == 4 or len(rotation) == 3 or len(rotation) == 9
        if len(rotation) == 4:
            rotation = pyrr.Quaternion(rotation)
        if len(rotation) == 3 or len(rotation) == 9:
            rotation = pyrr.Matrix33(rotation).quaternion

        if opencv2opengl:
            position, rotation = opencv_2_opengl(position, rotation)

        log.info(f"translation loaded: {position}")
        log.info(f"rotation loaded as quaternion: {rotation}")

        self._position = position
        self._rotation = rotation

        if self.qx is None:
            device = "cpu"
        else:
            device = self.qx.device
        self.qx = torch.nn.Parameter(torch.ones(batchsize) * rotation[0])
        self.qy = torch.nn.Parameter(torch.ones(batchsize) * rotation[1])
        self.qz = torch.nn.Parameter(torch.ones(batchsize) * rotation[2])
        self.qw = torch.nn.Parameter(torch.ones(batchsize) * rotation[3])

        self.x = torch.nn.Parameter(torch.ones(batchsize) * position[0])
        self.y = torch.nn.Parameter(torch.ones(batchsize) * position[1])
        self.z = torch.nn.Parameter(torch.ones(batchsize) * position[2])

        self.to(device)
        if not self.mesh is None:
            self.mesh.cuda()

    def set_batchsize(self, batchsize: int):
        """
        Change the batchsize to a new value, use the latest position and rotation to reset the batch of poses with. Be careful to make sure the image data is also updated accordingly.

        Args:
            batchsize (int): Batchsize to optimize
        """
        device = self.qx.device

        self.qx = torch.nn.Parameter(torch.ones(batchsize) * self._rotation[0])
        self.qy = torch.nn.Parameter(torch.ones(batchsize) * self._rotation[1])
        self.qz = torch.nn.Parameter(torch.ones(batchsize) * self._rotation[2])
        self.qw = torch.nn.Parameter(torch.ones(batchsize) * self._rotation[3])

        self.x = torch.nn.Parameter(torch.ones(batchsize) * self._position[0])
        self.y = torch.nn.Parameter(torch.ones(batchsize) * self._position[1])
        self.z = torch.nn.Parameter(torch.ones(batchsize) * self._position[2])

        self.to(device)

        if not self.mesh is None:
            self.mesh.set_batchsize(batchsize=batchsize)
            self.mesh.cuda()

    def __repr__(self):
        # TODO use the function for the argmax
        return f"Object3D( \n (pos): {self.x.shape} ,[0]:[{self.x[0].item(),self.y[0].item(),self.z[0].item()}] on {self.x.device}\n (mesh): {self.mesh} on {self.mesh.pos.device} \n)"

    def cuda(self):
        """
        not sure why I need to wrap this, but I had to for the mesh information
        """
        super().cuda()

        self.mesh.cuda()

    def reset_pose(self):
        """
        Reset the pose to what was passed during init. This could be called if an optimization was ran multiple times.
        """
        device = self.qx.device

        self.qx = torch.nn.Parameter(torch.ones(self.qx.shape[0]) * self._rotation[0])
        self.qy = torch.nn.Parameter(torch.ones(self.qx.shape[0]) * self._rotation[1])
        self.qz = torch.nn.Parameter(torch.ones(self.qx.shape[0]) * self._rotation[2])
        self.qw = torch.nn.Parameter(torch.ones(self.qx.shape[0]) * self._rotation[3])

        self.x = torch.nn.Parameter(torch.ones(self.qx.shape[0]) * self._position[0])
        self.y = torch.nn.Parameter(torch.ones(self.qx.shape[0]) * self._position[1])
        self.z = torch.nn.Parameter(torch.ones(self.qx.shape[0]) * self._position[2])

        self.cuda()

    def forward(self):
        """
        Return:
            returns a dict with field quat, trans.
        """
        q = torch.stack([self.qx, self.qy, self.qz, self.qw], dim=0).T
        q = q / torch.norm(q, dim=1).reshape(-1, 1)

        # TODO add the dict from object3d to the output of the module.
        to_return = self.mesh()
        to_return["quat"] = q
        to_return["trans"] = torch.stack([self.x, self.y, self.z], dim=0).T

        return to_return


@dataclass
class Image:
    """
    A class to represent a image, this could be a depth image or a rgb image, etc.

    *The image has to be upside down to work in DIFF-DOPE* so the image is flipped automatically, but if you initialize it yourself you should flip it.

    Args:
        img_path (str): a path to an image to load
        img_resize (float): bounded [0,1] to resize the image
        flip_img (bool): Default is True, when initialized to the image need to be flipped (diff-dope works with flipped images)
        img_tensor (torch.tensor): an image in tensor format, assumes the image is bounded [0,1]
    """

    img_path: Optional[str] = None
    img_tensor: Optional[torch.tensor] = None
    img_resize: Optional[float] = 1
    flip_img: Optional[bool] = True
    depth: Optional[bool] = False
    depth_scale: Optional[float] = 100

    def __post_init__(self):
        if not self.img_path is None:
            if self.depth:
                im = cv2.imread(self.img_path, cv2.IMREAD_UNCHANGED) / self.depth_scale
            else:
                im = cv2.imread(self.img_path)[:, :, :3]
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                # normalize
                im = im / 255.0
            if self.flip_img:
                im = cv2.flip(im, 0)

            if self.img_resize < 1.0:
                if self.depth:
                    im = cv2.resize(
                        im,
                        (
                            int(im.shape[1] * self.img_resize),
                            int(im.shape[0] * self.img_resize),
                        ),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    im = cv2.resize(
                        im,
                        (
                            int(im.shape[1] * self.img_resize),
                            int(im.shape[0] * self.img_resize),
                        ),
                    )
            self.img_tensor = torch.tensor(im).float()
            log.info(f"Loaded image {self.img_path}, shape: {self.img_tensor.shape}")
        self._batchsize_set = False

    def __repr__(self):
        return f"{self.img_tensor.shape} @ {self.img_path} on {self.img_tensor.device}"

    def __str__(self):
        return f"{self.img_tensor.shape} @ {self.img_path} on {self.img_tensor.device}"

    def cuda(self):
        """
        Switch the img_tensor to cuda tensor
        """
        self.img_tensor = self.img_tensor.cuda().float()

    def set_batchsize(self, batchsize):
        """
        Change the batchsize for the image tensor

        Args:
            batchsize (int): batchsize for the tensor
        """
        if self._batchsize_set is False:
            self.img_tensor = torch.stack([self.img_tensor] * batchsize, dim=0)
            self._batchsize_set = True

        else:
            self.img_tensor = torch.stack([self.img_tensor[0]] * batchsize, dim=0)


@dataclass
class Scene:
    """
    This class witholds images for the optimization.

    Attributes:
        tensor_rgb (torch.tensor): (batchsize,w,h,3) image tensor
        tensor_segmentation (torch.tensor): (batchsize,w,h,3) segmentation with 3 channels to facilate the computation later on.
        tensor_depth (torch.tensor): (batchsize,w,h) depth tensor (cm)
        image_resize (float): [0,1] bounding the image resize in percentage.
    """

    path_img: Optional[str] = None
    path_depth: Optional[str] = None
    path_segmentation: Optional[str] = None
    image_resize: Optional[float] = None

    tensor_rgb: Optional[Image] = None
    tensor_depth: Optional[Image] = None
    tensor_segmentation: Optional[Image] = None

    def __post_init__(self):
        # load the images and store them correctly
        if not self.path_img is None:
            self.tensor_rgb = Image(self.path_img, img_resize=self.image_resize)
        if not self.path_depth is None:
            self.tensor_depth = Image(
                self.path_depth, img_resize=self.image_resize, depth=True
            )
        if not self.path_segmentation is None:
            self.tensor_segmentation = Image(
                self.path_segmentation, img_resize=self.image_resize
            )

    def set_batchsize(self, batchsize):
        """
        Change the batchsize for the image tensors

        Args:
            batchsize (int): batchsize for the tensors
        """
        if not self.path_img is None:
            self.tensor_rgb.set_batchsize(batchsize)
        if not self.path_depth is None:
            self.tensor_depth.set_batchsize(batchsize)
        if not self.path_segmentation is None:
            self.tensor_segmentation.set_batchsize(batchsize)

    def get_resolution(self):
        """
        Get the scene image resolution for rendering

        Return
            (list): w,h of the image for optimization
        """
        if not self.path_img is None:
            return [
                self.tensor_rgb.img_tensor.shape[-3],
                self.tensor_rgb.img_tensor.shape[-2],
            ]
        if not self.path_depth is None:
            return [
                self.tensor_depth.img_tensor.shape[-2],
                self.tensor_depth.img_tensor.shape[-1],
            ]
        if not self.path_segmentation is None:
            return [
                self.tensor_segmentation.img_tensor.shape[-3],
                self.tensor_segmentation.img_tensor.shape[-2],
            ]

    def cuda(self):
        """
        Put on cuda the image tensors
        """

        if not self.path_img is None:
            self.tensor_rgb.cuda()
        if not self.path_depth is None:
            self.tensor_depth.cuda()
        if not self.path_segmentation is None:
            self.tensor_segmentation.cuda()


@dataclass
class DiffDope:
    """
    The main class containing all the information needed to run a Diff-DOPE optimization.
    This file is mostly driven by a config file using hydra, see the `configs/` folder.

    Args:
        cfg (DictConfig): a config file that populates the right information in the class. Please see `configs/diffdope.yaml` for more information.

    Attributes:
        optimization_results (list): a list of the different outputs from the optimization.
            Each entry is an optimization step.
            For an entry the keys are `{'rgb','depth','losses'}`
        gt_tensors (dict): a dict for `{'rgb','depth','segmentation'}' to access the torch tensor directly.
            This is useful for the image generation and for the losses defined by users.
            Moreover extent this so you can render your special losses. See examples.
        loss_functions (list): a list of function for the losses to be computed. See examples for how you could write yours.
        losses_values (dict): losses value per loss term and per image.
    """

    cfg: [DictConfig] = None
    camera: Optional[Camera] = None
    object3d: Optional[Object3D] = None
    scene: Optional[Scene] = None
    resolution: Optional[list] = None
    batchsize: Optional[int] = 16

    # TODO:
    # storing the renders for the optimization
    # how to pass add_loss
    # how to store losses
    # driven by the cfg

    def __post_init__(self):
        if self.camera is None:
            # load the camera from the config
            self.camera = Camera(**self.cfg.camera)
        # print(self.pose.position)
        if self.object3d is None:
            self.object3d = Object3D(**self.cfg.object3d)
        if self.scene is None:
            self.scene = Scene(**self.cfg.scene)
        self.batchsize = self.cfg.hyperparameters.batchsize

        # load the rendering
        self.glctx = dr.RasterizeGLContext()
        self.cuda()

        self.resolution = self.scene.get_resolution()

        self.optimization_results = []

        self.gt_tensors = {}

        if self.scene.tensor_rgb is not None:
            self.gt_tensors["rgb"] = self.scene.tensor_rgb.img_tensor
        if self.scene.tensor_depth is not None:
            self.gt_tensors["depth"] = self.scene.tensor_depth.img_tensor
        if self.scene.tensor_segmentation is not None:
            self.gt_tensors["segmentation"] = self.scene.tensor_segmentation.img_tensor

        self.set_batchsize(self.batchsize)

        # Storing the values for losses display
        self.losses_values = {}

        self.loss_functions = []
        if self.cfg.losses.l1_rgb_with_mask:
            self.loss_functions.append(dd.l1_rgb_with_mask)
        if self.cfg.losses.l1_depth_with_mask:
            self.loss_functions.append(dd.l1_depth_with_mask)
        if self.cfg.losses.l1_mask:
            self.loss_functions.append(dd.l1_mask)

        # self.object3d.mesh.enable_gradients_texture()

        # logging
        log.info(f"batchsize is {self.batchsize}")
        log.info(self.object3d)
        log.info(self.scene)

    def set_batchsize(self, batchsize):
        """
        Set the batchsize for the optimization

        Args
        batchsize (int): batchsize format
        """
        self.batchsize = batchsize
        self.scene.set_batchsize(batchsize)
        self.object3d.set_batchsize(batchsize)
        self.camera.set_batchsize(batchsize)

        # todo add a cfg call here.
        # self.object3d.mesh.enable_gradients_texture()

        self.optimizer = torch.optim.SGD(
            self.object3d.parameters(), lr=self.cfg.hyperparameters.learning_rate_base
        )

        # TODO add a seed to the random
        self.learning_rates = [
            random.uniform(
                self.cfg.hyperparameters.learning_rates_bound[0],
                self.cfg.hyperparameters.learning_rates_bound[1],
            )
            for _ in range(batchsize)
        ]
        self.learning_rates = torch.tensor(self.learning_rates).float().cuda()

    def render_img(
        self,
        index=None,
        batch_index=None,
        render_selection="rgb",
    ):
        """
        Rendering an image with the render overlay on the image or not.
            Check config `render_images` entry for hyper params for this function.

        Parameters:
            nrow (int): grid row width
            final_width_batch (int): grid final width in pixel
            add_background (bool): do you want to have the image we optimize as
                background?
            alpha_overlay (float): The transparency applied to the rendered 3d model
                so you can see how it overlays
            crop_around_mask (bool): Crop around the provided mask, if no mask then uses the render to crop, but crop will be moving

        Args:
            index (int): None, which index of the optimization you want to render
                if None, renders the last one
            batch_index (int): None, which batch indez you want to render as image
                If None, renders all the images in the batch.
            render_selection (str): 'rgb','depth',

        Returns:
            An image in the form of a nd.array (cv2 style)
        """

        # TODO
        # check if there is an argmin and render that one instead of none.

        if index is None:
            index = -1
        else:
            assert index < len(self.optimization_results) and index >= 0

        if self.cfg.render_images.crop_around_mask:
            if "segmentation" in self.gt_tensors.keys():
                crop = find_crop(self.gt_tensors["segmentation"][0])

            else:
                crop = find_crop(self.optimization_results[index][render_selection][0])
        
        if batch_index is None:
            # make a grid
            if self.cfg.render_images.crop_around_mask:
                gt_tensor = self.gt_tensors[render_selection][
                    :,
                    crop[0] : crop[0] + crop[2] + 1,
                    crop[1] : crop[1] + crop[2] + 1,
                    ...,
                ]
                gu_tensor = self.optimization_results[index][render_selection][
                    :,
                    crop[0] : crop[0] + crop[2] + 1,
                    crop[1] : crop[1] + crop[2] + 1,
                    ...,
                ]
            else:
                gt_tensor = self.gt_tensors[render_selection]
                gu_tensor = self.optimization_results[index][render_selection]

            img = make_grid_overlay_batch(
                background=gt_tensor,
                foreground=gu_tensor,
                alpha=self.cfg.render_images.alpha_overlay,
                row=self.cfg.render_images.nrow,
                final_width=self.cfg.render_images.final_width_batch,
                add_background=self.cfg.render_images.add_background,
                add_contour=self.cfg.render_images.add_countour,
                color_countour=self.cfg.render_images.color_countour,
                flip_result=self.cfg.render_images.flip_result,
            )

            return img
        else:
            if self.cfg.render_images.crop_around_mask:
                gt_tensor = self.gt_tensors[render_selection][
                    batch_index,
                    crop[0] : crop[0] + crop[2] + 1,
                    crop[1] : crop[1] + crop[2] + 1,
                    ...,
                ]
                gu_tensor = self.optimization_results[index][render_selection][
                    batch_index,
                    crop[0] : crop[0] + crop[2] + 1,
                    crop[1] : crop[1] + crop[2] + 1,
                    ...,
                ]
            else:
                gt_tensor = self.gt_tensors[render_selection][batch_index]
                gu_tensor = self.optimization_results[index][render_selection][
                    batch_index
                ]

            img = make_grid_overlay_batch(
                background=gt_tensor.unsqueeze(0),
                foreground=gu_tensor.unsqueeze(0),
                alpha=self.cfg.render_images.alpha_overlay,
                row=self.cfg.render_images.nrow,
                final_width=self.cfg.render_images.final_width_batch,
                add_background=self.cfg.render_images.add_background,
                add_contour=self.cfg.render_images.add_countour,
                color_countour=self.cfg.render_images.color_countour,
                flip_result=self.cfg.render_images.flip_result,
            )

        return img

    def get_argmin(self):
        """
        Returns the argmin of all the losses put together
        """
        # Calculate the average tensor across all keys at the last time step
        last_time_step = -1  # Index for the last time step

        # average_tensor = torch.stack([tensor[last_time_step] for tensor in self.losses_values.values()]).mean(dim=0)

        tensor_list = []

        # Extract tensors at the last time step for each key and add to the list
        for key, tensor in self.losses_values.items():
            tensor_at_last_time_step = tensor[last_time_step]
            tensor_list.append(tensor_at_last_time_step)

        # Stack the list of tensors along dimension 0 to create a single tensor
        stacked_tensor = torch.stack(tensor_list, dim=0)

        # Calculate the mean along dimension 0 to get the average tensor
        average_tensor = stacked_tensor.mean(dim=0)

        # Find the argmin of the average tensor
        argmin = torch.argmin(average_tensor, dim=-1)

        return argmin

    def make_animation(self, output_file_path=None, frame_rate=20, batch_index=-1):
        """
        Make an animation of the optimization to be saved. This uses the `render_img` function.

        Args:
            output_file_path (str): output path, if None, this goes into the hydra tmp folder.
            frame_rate (int): video frame rate
            batch_index (int): if -1 use argmin function.
        """

        if output_file_path is None:
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            output_file_path = f"{hydra_cfg['runtime']['output_dir']}/animation.mp4"

        # Set the frame rate (change this value as needed)
        frame_rate = 10

        # Get the dimensions of the first image (assuming all images have the same dimensions)
        height, width = self.resolution

        # Create a VideoWriter object to save the MP4 video (use 'XVID' codec for MP4)
        writer = imageio.get_writer(
            output_file_path, mode="I", fps=frame_rate, codec="libx264", bitrate="16M"
        )

        if batch_index == -1:
            batch_index = self.get_argmin()

        # Loop through the list of images and add each frame to the video
        pbar = tqdm(range(self.cfg.hyperparameters.nb_iterations + 1))
        for iteration_now in pbar:
            pbar.set_description("making video")
            # Ensure the image is in BGR format (OpenCV default)
            img = self.render_img(index=iteration_now, batch_index=batch_index)
            writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Release the VideoWriter to save the MP4 video
        writer.close()

    def add_loss_value(self, key, values, values_weighted=None):
        """
        Store in `losses_values` the values of different loss term at for all the objects
            This function adds to the list of values, thus if you call multiple times on the same key, values are just added and not checked.
        Args:
            key (str): key to be store in the dict
            values (torch.tensor): B of size to be stored at key
        """

        if key not in self.losses_values:
            self.losses_values[key] = (
                values.detach().cpu().unsqueeze(0)
            )  # Create an empty tensor of the same data type
        else:
            # Append the new values (assuming values is a 1D tensor)
            self.losses_values[key] = torch.cat(
                (self.losses_values[key], values.detach().cpu().unsqueeze(0)), dim=0
            )

    def plot_losses(self, keys=None, batch_index=-1):
        """
        Plot the losses value

        Args:
            keys (list(str)): keys you want to plot, if None, all are plotted.
            including_weighting (bool): do you want to include the weighting
            batch_index (int): if -1 returns the argmin uses.

        Returns:
            returns a nd.array as image

        """

        if len(self.losses_values.keys()) == 0:
            return None

        if batch_index == -1:
            batch_index = self.get_argmin()

        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

        # Plot the batch_index values as lines
        for i, key in enumerate(self.losses_values.keys()):
            plt.plot(
                self.losses_values[key][..., batch_index].numpy(), marker="o", label=key
            )
        # plt.show()
        plt.legend()

        buffer = io.BytesIO()

        # Save the plot to the buffer instead of a file
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)

        # Convert the PIL image to a NumPy array
        pil_image = pilImage.open(buffer)
        numpy_array = np.array(pil_image)
        image_rgb = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2RGB)
        # Close the plot to release resources
        plt.close()

        return image_rgb

    def get_pose(self, batch_index=-1):
        """
        return the matrix pose as a np.ndarray

        Args:
            batch_index (int): if -1 use argmin function.

        Returns a 4x4 np.ndarray
        """
        if batch_index == -1:
            batch_index = self.get_argmin()

        matrix44 = self.optimization_results[-1]["mtx"][batch_index].numpy()

        return matrix44

    def run_optimization(self):
        """
        If the class is set correctly this runs the optimization for finding a good pose
        """

        self.losses_values = {}
        self.optimization_results = []

        self.optimizer = torch.optim.SGD(
            self.object3d.parameters(), lr=self.cfg.hyperparameters.learning_rate_base
        )

        if self.scene.tensor_rgb is not None:
            self.gt_tensors["rgb"] = self.scene.tensor_rgb.img_tensor
        if self.scene.tensor_depth is not None:
            self.gt_tensors["depth"] = self.scene.tensor_depth.img_tensor
        if self.scene.tensor_segmentation is not None:
            self.gt_tensors["segmentation"] = self.scene.tensor_segmentation.img_tensor


        pbar = tqdm(range(self.cfg.hyperparameters.nb_iterations + 1))

        for iteration_now in pbar:
            itf = iteration_now / self.cfg.hyperparameters.nb_iterations + 1
            lr = (
                self.cfg.hyperparameters.base_lr
                * self.cfg.hyperparameters.lr_decay**itf
            )

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            self.optimizer.zero_grad()

            result = self.object3d()

            # transform quat and position into a matrix44
            mtx_gu = matrix_batch_44_from_position_quat(
                p=result["trans"], q=result["quat"]
            )

            if self.object3d.mesh.has_textured_map is False:
                self.renders = render_texture_batch(
                    glctx=self.glctx,
                    proj_cam=self.camera.cam_proj,
                    mtx=mtx_gu,
                    pos=result["pos"],
                    pos_idx=result["pos_idx"],
                    vtx_color=result["vtx_color"],
                    resolution=self.resolution,
                )
            else:
                # TODO test the index color version
                self.renders = render_texture_batch(
                    glctx=self.glctx,
                    proj_cam=self.camera.cam_proj,
                    mtx=mtx_gu,
                    pos=result["pos"],
                    pos_idx=result["pos_idx"],
                    uv=result["uv"],
                    uv_idx=result["uv_idx"],
                    tex=result["tex"],
                    resolution=self.resolution,
                )
            to_add = {}
            to_add["rgb"] = self.renders["rgb"].detach().cpu()
            to_add["depth"] = self.renders["depth"].detach().cpu()
            to_add["mtx"] = mtx_gu.detach().cpu()

            self.optimization_results.append(to_add)

            # computing the losses
            loss = torch.zeros(1).cuda()
            for loss_function in self.loss_functions:
                l = loss_function(self)
                if l is None:
                    continue
                loss += l
            pbar.set_description(f"loss: {loss.item():.4f}")
            loss.backward()
            self.optimizer.step()

    def cuda(self):
        """
        Copy variables to the GPU.
        """
        # check the projection matrix
        #
        self.object3d.cuda()
        self.scene.cuda()
        self.camera.cuda()
        pass
