import os
import sys

import rospkg

diffdope_ros_path = rospkg.RosPack().get_path("diffdope_ros")
sys.path.insert(0, os.path.join(diffdope_ros_path, "scripts"))

import actionlib
import cv2
import hydra
import rospy
import torch
from cv_bridge import CvBridge
from diffdope_ros.msg import (
    RefineAllAction,
    RefineAllActionResult,
    RefineObjectAction,
    RefineObjectActionResult,
    TargetObject,
)
from icecream import ic
from omegaconf import DictConfig
from segmentator import SegmentAnything

import diffdope as dd


class DiffDOPEServer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        checkpoint_path = os.path.expanduser(self.cfg.segment_anything_checkpoint_path)
        self.segment_anything = SegmentAnything(self.cfg.camera, checkpoint_path)

        self._action_refine_object_server = actionlib.SimpleActionServer(
            "refine_object",
            RefineObjectAction,
            execute_cb=self.refine_object,
            auto_start=False,
        )

        self._action_refine_all_server = actionlib.SimpleActionServer(
            "refine_all", RefineAllAction, execute_cb=self.refine_all, auto_start=False
        )

        self._action_refine_object_server.start()
        self._action_refine_all_server.start()
        rospy.loginfo(
            "[DiffDOPE Server] Both action servers started. Waiting for requests..."
        )

    def refine_object(self, goal):
        result = self._compute_refined_pose(
            goal.target_object, RefineObjectActionResult()
        )
        self._action_refine_object_server.set_succeeded(result)

    def refine_all(self, target_objects):
        for target_object in target_objects:
            result = self._compute_refined_pose(
                target_object, RefineObjectActionResult()
            )
            self._action_refine_all_server.set_succeeded(result)

    def _create_pose_stamped(self, diffdope_pose):
        return None

    def _compute_refined_pose(self, target_object, result):
        initial_pose_estimate = target_object.pose.pose
        scene = self._generate_scene(
            target_object.rgb_frame,
            target_object.depth_frame,
            initial_pose_estimate.position,
        )
        object3d = self._generate_3d_object(target_object)

        ddope = dd.DiffDope(cfg=self.cfg, scene=scene, object3d=object3d)
        ddope.run_optimization()
        print(ddope.get_pose())
        ddope.make_animation(output_file_path=os.path.expanduser("~/simple_scene.mp4"))

        refined_pose = self._create_pose_stamped(ddope.get_pose())
        target_object = TargetObject()
        target_object.name = target_object.name
        target_object.pose = refined_pose
        # result.refined_estimation = target_object

        return result

    def _generate_scene(self, rgb_frame, depth_frame, point_in_camera):
        bridge = CvBridge()
        try:
            cv_image = bridge.imgmsg_to_cv2(rgb_frame, "bgr8")
            rgb_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            output_path = (
                "/home/rafael/debug"  # os.path.join(diffdope_ros_path, 'debug')
            )
            cv2.imwrite(os.path.join(output_path, "rgb1.png"), cv_image)
            cv2.imwrite(os.path.join(output_path, "rgb2.png"), rgb_frame)

            segmentation_frame = self.segment_anything.segment(
                rgb_frame, point_in_camera
            )

            rgb_frame_normalized = rgb_frame / 255.0
            rgb_frame = cv2.flip(rgb_frame_normalized, 0)  # needed by diff-dope
            cv2.imwrite(os.path.join(output_path, "rgb3.png"), rgb_frame)
        except Exception as e:
            rospy.logerr(f"CvBridge Error: {e}")

        depth_scale = 100
        try:
            depth_frame = bridge.imgmsg_to_cv2(depth_frame, "passthrough") / depth_scale
            depth_frame = cv2.flip(depth_frame, 0)  # needed by diff-dope
        except Exception as e:
            rospy.logerr(f"CvBridge Error: {e}")

        rgb_tensor = torch.tensor(rgb_frame).float()
        rgb_image = dd.Image(img_tensor=rgb_tensor, img_resize=self.cfg.image_resize)

        depth_tensor = torch.tensor(depth_frame).float()
        depth_image = dd.Image(
            img_tensor=depth_tensor, img_resize=self.cfg.image_resize, depth=True
        )

        # segmentation_frame = segmentation_frame / 255.0
        # segmentation_frame = cv2.flip(
        # segmentation_frame, 0
        # )  # needed by diff-dope

        output_path = "/home/rafael/debug"  # os.path.join(diffdope_ros_path, 'debug')
        cv2.imwrite(os.path.join(output_path, "rgb.png"), rgb_frame)
        # cv2.imwrite(os.path.join(output_path, 'depth.png'), depth_frame)
        cv2.imwrite(os.path.join(output_path, "seg.png"), segmentation_frame)
        # segmentation_tensor = torch.tensor(segmentation_frame).float()

        # segmentation_image = dd.Image(
        #     img_tensor=segmentation_tensor, img_resize=self.cfg.image_resize
        # )

        scene = dd.Scene(
            tensor_rgb=rgb_image,
            tensor_depth=depth_image,
            path_segmentation=output_path + "/seg.png",
            image_resize=1.0
            # tensor_segmentation=segmentation_image,
        )

        return scene

    def _generate_3d_object(self, goal: TargetObject):
        pos = goal.pose.pose.position
        position = [pos.x * 1_000, pos.y * 1_000, pos.z * 1_000]

        quat = goal.pose.pose.orientation
        rotation = [quat.x, quat.y, quat.z, quat.w]  # respecting diffdope order

        model_path = os.path.join(diffdope_ros_path, goal.model_path)
        object3d = dd.Object3D(
            position, rotation, model_path=model_path  # , scale=goal.scale
        )

        return object3d


def parse_cfg():
    diffdope_ros_path = rospkg.RosPack().get_path("diffdope_ros")
    config_dir = os.path.join(diffdope_ros_path, "configs")
    config_file = sys.argv[1]

    hydra.initialize_config_dir(version_base=None, config_dir=config_dir)
    cfg = hydra.compose(config_name=config_file, return_hydra_config=True)

    return cfg


if __name__ == "__main__":
    rospy.init_node("diffdope_action_server")

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        rospy.loginfo(f"[DiffDOPE Server] Config file working with: {config_file}")
    else:
        sys.exit("No config file provided.")

    cfg = parse_cfg()
    server = DiffDOPEServer(cfg)
    rospy.spin()
