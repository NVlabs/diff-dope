import os
import rospkg
from collections import namedtuple
import torch
import numpy as np
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import SamPredictor


class SegmentAnything:
    def __init__(self, camera_parameters, checkpoint_path):
        self.camera_parameters = camera_parameters
        self.sam = sam_model_registry['vit_b'](checkpoint=checkpoint_path)

        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        _ = self.sam.to(device=DEVICE)

        self.mask_predictor = SamPredictor(self.sam)

    def _project_position_to_image_plane(self, position):
        x = position[0] / position[2]
        y = position[1] / position[2]

        new_x = self.camera_parameters.fx * x + self.camera_parameters.cx
        new_y = self.camera_parameters.fy * y + self.camera_parameters.cy

        return new_x, new_y

    def segment(self, image_bgr, point_in_camera):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.mask_predictor.set_image(image_rgb)
        x, y = self._project_position_to_image_plane(point_in_camera)

        masks, _, _ = self.mask_predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True
        )

        largest_mask = max(masks, key=np.count_nonzero)

        return largest_mask


def dummy_demo():
    diffdope_ros_path = rospkg.RosPack().get_path('diffdope_ros')
    input_dir = os.path.join(diffdope_ros_path, "example_data", "segmentation_inputs")

    DOPEPoint = namedtuple('DOPEPoint', 'x y z')
    data = [
        (os.path.join(input_dir, 'test1.png'), DOPEPoint(-0.09592835017609432, 0.08744199450987605, 0.5177931977536219)),
        (os.path.join(input_dir, 'test2.png'), DOPEPoint(-0.0658483455855903, 0.03527420196672081, 0.4323496926767813)),
    ]

    Camera = namedtuple('Camera', 'fx fy cx cy')
    camera = Camera(fx=908.85, fy=906.69, cx=626.71, cy=383.20)
    sa = SegmentAnything(camera, os.path.expanduser('~/sam_vit_b_01ec64.pth'))

    output_dir = os.path.join(diffdope_ros_path, 'example_data', 'segmentation_output')

    for i, (filename, dope_position) in enumerate(data):
        print(f'Segmenting {filename}')
        image_bgr = cv2.imread(f'{filename}')
        segmentation_mask = sa.segment(image_bgr, dope_position)

        path = os.path.join(output_dir, f"output{i}.png")
        cv2.imwrite(path, segmentation_mask * 255)
        print(f'Output segmentation saved at {path}')
        print()

if __name__ == "__main__":
    dummy_demo()
