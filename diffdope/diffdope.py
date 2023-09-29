import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import hydra
import numpy as np
import pyrr
import torch
import trimesh
from omegaconf import DictConfig, OmegaConf

# A logger for this file
log = logging.getLogger(__name__)


def dist_batch_lr(tensor, learning_rates, channels=[1, 2, 3]):
    """
    Method that distribute different learning rates to the batch.

    Args:
        tensor (torch.tensor): BxWxHxC if you do not have C, pass a different set of channel, for example you could run on a depth map [1,2].
        learning_rates (torch.tensor): B the different learning rates needed.
        channels (list): the index values used to apply the first mean, e.g., [1,2,3] for colored image, or [1,2] for a depth map
    """

    return torch.mean((torch.mean(tensor, channels) * learning_rates))


def opencv_2_opengl(p, q):
    """
    Converting the pose from opencv coordinate to opengl coordinate

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


@dataclass
class Mesh:
    """
    A wrapper around a Trimesh mesh, where the data is already loaded
    to be consumed by PyTorch. As such the internal values are stored as
    torch array.

    Args:
        path_model (str): path to the object to be loaded, see Trimesh which extensions are supported.

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

    path_model: str

    def __post_init__(self):
        # load the mesh
        mesh = trimesh.load(self.path_model, force="mesh")

        pos = np.asarray(mesh.vertices)
        pos_idx = np.asarray(mesh.faces)

        normals = np.asarray(mesh.vertex_normals)

        pos_idx = torch.from_numpy(pos_idx.astype(np.int32))

        #
        # TODO THIS USED TO HAVE /100 on the vertex_pos move to a general scale function.
        #

        vtx_pos = torch.from_numpy(pos.astype(np.float32))
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
            tex = np.array(mesh.visual.material.image)
            uv = mesh.visual.uv
            uv[:, 1] = 1 - uv[:, 1]
            uv_idx = np.asarray(mesh.faces)

            tex = np.array(mesh.visual.material.image) / 255.0

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
            self.textured_map = True
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
            self.textured_map = False

        log.info(
            f"loaded mesh @{self.path_model}. Does it have texture map? {self.textured_map} "
        )


class Pose(torch.nn.Module):
    """
    This is the batch pose representation that Diff-DOPE uses to optimize.

    Attributes:
        qx,qy,qz,qw (torch.nn.Parameter): batchsize x 1 representing the quaternion
        x,y,z (torch.nn.Parameter): batchsize x 1 representing the position
    """

    def __init__(
        self,
        position: list,
        rotation: list,
        batchsize: int = 32,
        opencv2opengl: bool = True,
    ):
        """
        Args:
            position (list): a 3 value list of the object position
            rotation (list): could be a quat with 4 values (x,y,z,w), or a flatten rotational matrix or a 3x3 matrix (as a list of list) -- both are row-wise / row-major.
            batchsize (int): size of the batch to be optimized, this is defined normally as a hyperparameter.
            opencv2opengl (bool): converting the coordinate space from one to the other
        """
        super().__init__()
        self.qx = None  # to load on cpu and not gpu

        self.set_pose(position, rotation, batchsize, opencv2opengl=opencv2opengl)

    def set_pose(
        self,
        position: list,
        rotation: list,
        batchsize: int = 32,
        opencv2opengl: bool = True,
    ):
        """
        Set the pose to new values, the inputs can be either list, numpy or torch.tensor. If the class was put on cuda(), the updated pose should be on the GPU as well.

        Args:
            position (list): a 3 value list of the object position
            rotation (list): could be a quat with 4 values (x,y,z,w), or a flatten rotational matrix or a 3x3 matrix (as a list of list) -- both are row-wise / row-major.
            batchsize (int): size of the batch to be optimized, this is defined normally as a hyperparameter.
        """

        assert len(position) == 3
        position = np.array(position)

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
        self.qx = torch.nn.Parameter(torch.ones(batchsize) * rotation[0]).to(device)
        self.qy = torch.nn.Parameter(torch.ones(batchsize) * rotation[1]).to(device)
        self.qz = torch.nn.Parameter(torch.ones(batchsize) * rotation[2]).to(device)
        self.qw = torch.nn.Parameter(torch.ones(batchsize) * rotation[3]).to(device)

        self.x = torch.nn.Parameter(torch.ones(batchsize) * position[0]).to(device)
        self.y = torch.nn.Parameter(torch.ones(batchsize) * position[1]).to(device)
        self.z = torch.nn.Parameter(torch.ones(batchsize) * position[2]).to(device)

    def set_batchsize(self, batchsize: int):
        """
        Change the batchsize to a new value, use the latest position and rotation to reset the batch of poses with. Be careful to make sure the image data is also updated accordingly.

        Args:
            batchsize (int): Batchsize to optimize
        """
        device = self.qx.device

        self.qx = torch.nn.Parameter(torch.ones(batchsize) * self._rotation[0]).to(
            device
        )
        self.qy = torch.nn.Parameter(torch.ones(batchsize) * self._rotation[1]).to(
            device
        )
        self.qz = torch.nn.Parameter(torch.ones(batchsize) * self._rotation[2]).to(
            device
        )
        self.qw = torch.nn.Parameter(torch.ones(batchsize) * self._rotation[3]).to(
            device
        )

        self.x = torch.nn.Parameter(torch.ones(batchsize) * self._position[0]).to(
            device
        )
        self.y = torch.nn.Parameter(torch.ones(batchsize) * self._position[1]).to(
            device
        )
        self.z = torch.nn.Parameter(torch.ones(batchsize) * self._position[2]).to(
            device
        )
        pass

    def reset_pose(self):
        """
        Reset the pose to what was passed during init. This could be called if an optimization was ran multiple times.
        """
        device = self.qx.device

        self.qx = torch.nn.Parameter(
            torch.ones(self.qx.shape[0]) * self._rotation[0]
        ).to(device)
        self.qy = torch.nn.Parameter(
            torch.ones(self.qx.shape[0]) * self._rotation[1]
        ).to(device)
        self.qz = torch.nn.Parameter(
            torch.ones(self.qx.shape[0]) * self._rotation[2]
        ).to(device)
        self.qw = torch.nn.Parameter(
            torch.ones(self.qx.shape[0]) * self._rotation[3]
        ).to(device)

        self.x = torch.nn.Parameter(
            torch.ones(self.qx.shape[0]) * self._position[0]
        ).to(device)
        self.y = torch.nn.Parameter(
            torch.ones(self.qx.shape[0]) * self._position[1]
        ).to(device)
        self.z = torch.nn.Parameter(
            torch.ones(self.qx.shape[0]) * self._position[2]
        ).to(device)

        pass

    def forward(self):
        """
        Return:
            returns a dict with field quat, trans.
        """

        q = torch.stack([self.qx, self.qy, self.qz, self.qw], dim=0).T
        q = q / torch.norm(q, dim=1).reshape(-1, 1)

        return {
            "quat": q,
            "trans": torch.stack([self.x, self.y, self.z], dim=0).T,
        }


@dataclass
class Image:
    """
    A class to represent a image, this could be a depth image or a rgb image, etc.

    *The image has to be upside down to work in DIFF-DOPE*

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
            self.img_tensor = torch.tensor(im).float() / 255.0
            log.info(f"Loaded image {self.img_path}, shape: {self.img_tensor.shape}")

    def cuda(self):
        """
        Switch the img_tensor to cuda tensor
        """
        self.img_tensor = self.img_tensor.cuda()


@dataclass
class DiffDope:
    """
    The main class containing all the information needed to run a Diff-DOPE optimization.
    This file is mostly driven by a config file using hydra, see the `configs/` folder.

    Args:
        cfg (DictConfig): a config file that populates the right information in the class. Please see `configs/diffdope.yaml` for more information.
    """

    cfg: Optional[DictConfig] = None
    camera: Optional[Camera] = None
    pose: Optional[Pose] = None
    mesh: Optional[Mesh] = None

    # TODO:
    # resize function for the image?
    # storing the images
    # storing the renders for the optimization
    # how to pass add_loss
    # how to store losses
    # driven by the cfg

    def __post_init__(self):
        if not self.cfg is None:
            if self.camera is None:
                # load the camera from the config
                self.camera = Camera(**self.cfg.camera)
            # print(self.pose.position)
            if self.pose is None:
                self.pose = Pose(**self.cfg.pose)
            if self.mesh is None:
                self.mesh = Mesh(self.cfg.scene_path.model_path)

    def cuda(self):
        """
        Copy variables to the GPU.
        """
        # check the projection matrix
        #
        pass
