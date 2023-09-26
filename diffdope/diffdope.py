import numpy as np

import trimesh
import logging
import torch 
import hydra
import pyrr

from dataclasses import dataclass
from typing import Optional
from omegaconf import DictConfig, OmegaConf


# A logger for this file
log = logging.getLogger(__name__)

@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    im_width: int
    im_height: int
    znear: Optional[float] = 0.01
    zfar: Optional[float] = 200

    def get_projection_matrix(self):
        """Conversion of Hartley-Zisserman intrinsic matrix to OpenGL proj. matrix.

        Ref:
        1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
        2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py
        3) https://github.com/megapose6d/megapose6d/blob/3f5b51d9cef71d9ac0ac36c6414f35013bee2b0b/src/megapose/panda3d_renderer/types.py
        """

        K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        x0 = 0
        y0 = 0
        w = self.im_width
        h = self.im_height
        nc = self.znear
        fc = self.zfar

        window_coords='y_down'

        depth = float(fc - nc)
        q = -(fc + nc) / depth
        qn = -2 * (fc * nc) / depth

        # Draw our images upside down, so that all the pixel-based coordinate
        # systems are the same.
        if window_coords == 'y_up':
            proj = np.array([
            [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
            [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
            [0, 0, q, qn],  # Sets near and far planes (glPerspective).
            [0, 0, -1, 0]
            ])

        # Draw the images upright and modify the projection matrix so that OpenGL
        # will generate window coords that compensate for the flipped image coords.
        else:
            assert window_coords == 'y_down'
            proj = np.array([
            [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
            [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
            [0, 0, q, qn],  # Sets near and far planes (glPerspective).
            [0, 0, -1, 0]
            ])
        
        return proj

@dataclass
class Mesh:
    path_model: str
    
    def __post_init__(self):
        # load the mesh
        mesh = trimesh.load(
                self.path_model,
                force='mesh'
            )

        pos = np.asarray(mesh.vertices)
        pos_idx = np.asarray(mesh.faces)

        normals = np.asarray(mesh.vertex_normals)
        
        pos_idx = torch.from_numpy(pos_idx.astype(np.int32))
        vtx_pos = torch.from_numpy(pos.astype(np.float32))
        vtx_normals = torch.from_numpy(normals.astype(np.float32))
        bounding_volume = [
            [torch.min(vtx_pos[:,0]),torch.min(vtx_pos[:,1]),torch.min(vtx_pos[:,2])],
            [torch.max(vtx_pos[:,0]),torch.max(vtx_pos[:,1]),torch.max(vtx_pos[:,2])]
        ]

        dimensions = [
            bounding_volume[1][0] - bounding_volume[0][0], 
            bounding_volume[1][1] - bounding_volume[0][1], 
            bounding_volume[1][2] - bounding_volume[0][2]
        ]
        center_point = [
            ((bounding_volume[0][0] + bounding_volume[1][0])/2).item(), 
            ((bounding_volume[0][1] + bounding_volume[1][1])/2).item(), 
            ((bounding_volume[0][2] + bounding_volume[1][2])/2).item()
        ]

        if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
            tex = np.array(mesh.visual.material.image)
            uv = mesh.visual.uv
            uv[:,1] = 1 - uv[:,1]
            uv_idx = np.asarray(mesh.faces)

            tex = np.array(mesh.visual.material.image)/255.0

            tex     = torch.from_numpy(tex.astype(np.float32))
            uv_idx  = torch.from_numpy(uv_idx.astype(np.int32))
            vtx_uv  = torch.from_numpy(uv.astype(np.float32))    

            self.pos_idx =  pos_idx
            self.pos =  vtx_pos
            self.tex =  tex
            self.uv =  vtx_uv
            self.uv_idx =  uv_idx
            self.bounding_volume=bounding_volume
            self.dimensions=dimensions
            self.center_point=center_point
            self.vtx_normals=vtx_normals
            self.textured_map = True
        else:
            vertex_color = mesh.visual.vertex_colors[...,:3]/255.0
            vertex_color = torch.from_numpy(vertex_color.astype(np.float32))

            self.pos_idx =   pos_idx
            self.pos =   vtx_pos
            self.vtx_color =  vertex_color
            self.bounding_volume =  bounding_volume
            self.dimensions =  dimensions
            self.center_point =  center_point
            self.vtx_normals =  vtx_normals
            self.textured_map = False

        log.info(f'loaded mesh @{self.path_model}. Does it have texture map? {self.textured_map} ')



@dataclass
class Pose:
    position: list
    rotation: list

    def __post_init__(self):
        assert len(self.position) == 3
        self.position = np.array(self.position)

        assert len(self.rotation) == 4 or len(self.rotation) == 3 or len(self.rotation) == 9 
        if len(self.rotation) == 4:
            self.rotation = pyrr.Quaternion(self.rotation)   
        if len(self.rotation) == 3 or len(self.rotation) == 9:
            self.rotation = pyrr.Matrix33(self.rotation)   
        
        log.info(f'translation loaded: {self.position}')
        log.info(f'rotation loaded as quaternion: {self.rotation}')



@dataclass
class DiffDope:
    cfg: Optional[DictConfig]=None 
    camera: Optional[Camera]=None
    pose: Optional[Pose]=None
    mesh: Optional[Mesh]=None

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












