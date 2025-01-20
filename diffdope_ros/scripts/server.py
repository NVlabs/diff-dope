#!/bin/python3
import os
import re
import sys

import rospkg

diffdope_ros_path = rospkg.RosPack().get_path("diffdope_ros")
sys.path.insert(0, os.path.join(diffdope_ros_path, "scripts"))

from collections import namedtuple

import actionlib
import cv2
import hydra
import numpy as np
import rospy
import tf.transformations as tf_trans
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from omegaconf import DictConfig, OmegaConf, open_dict
from segmentator import SegmentAnything
from sensor_msgs.msg import CameraInfo

import diffdope as dd
from diffdope_ros.msg import (
    RefineAllAction,
    RefineAllResult,
    RefineObjectAction,
    RefineObjectResult,
    TargetObject,
)


class DiffDOPEServer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.num_of_saved_videos_in_this_run = 0
        self.camera_parameters = self.__get_camera_intrinsic_params(
            self.cfg.topics.camera_info
        )

        # diff-dope internally needs the camera parameters in self.cfg
        self.__add_camera_parameters_to_cfg()

        checkpoint_path = os.path.expanduser(self.cfg.segment_anything_checkpoint_path)
        self.segment_anything = SegmentAnything(self.camera_parameters, checkpoint_path)
        self.__advertise_actions()

    def __advertise_actions(self):
        self._action_refine_object_server = actionlib.SimpleActionServer(
            "refine_object",
            RefineObjectAction,
            execute_cb=self.__refine_object,
            auto_start=False,
        )

        self._action_refine_all_server = actionlib.SimpleActionServer(
            "refine_all",
            RefineAllAction,
            execute_cb=self.__refine_all,
            auto_start=False,
        )

        self._action_refine_object_server.start()
        self._action_refine_all_server.start()
        rospy.loginfo(
            "[DiffDOPE Server] Both action servers started. Waiting for requests..."
        )

    def __add_camera_parameters_to_cfg(self):
        OmegaConf.set_struct(self.cfg, True)
        with open_dict(self.cfg):
            self.cfg.camera = {}
            self.cfg.camera.fx = self.camera_parameters.fx
            self.cfg.camera.fy = self.camera_parameters.fy
            self.cfg.camera.cx = self.camera_parameters.cx
            self.cfg.camera.cy = self.camera_parameters.cy
            self.cfg.camera.im_width = self.camera_parameters.image_width
            self.cfg.camera.im_height = self.camera_parameters.image_height

    def __get_camera_intrinsic_params(self, camera_info_topic_name):
        camera_info = None

        def camera_info_callback(data):
            nonlocal camera_info
            camera_info = data

        rospy.Subscriber(camera_info_topic_name, CameraInfo, camera_info_callback)

        while camera_info is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        fx = camera_info.K[0]  # Focal length in x-axis
        fy = camera_info.K[4]  # Focal length in y-axis
        cx = camera_info.K[2]  # Principal point in x-axis
        cy = camera_info.K[5]  # Principal point in y-axis

        image_height = camera_info.height
        image_width = camera_info.width

        Camera = namedtuple("Camera", "fx fy cx cy image_height image_width")
        camera_intrinsic_parameters = Camera(fx, fy, cx, cy, image_height, image_width)
        return camera_intrinsic_parameters

    def __refine_object(self, goal):
        refined_pose = self.__compute_refined_pose(goal.target_object)
        result = RefineObjectResult()
        result.refined_estimation = refined_pose
        self._action_refine_object_server.set_succeeded(result)

    def __refine_all(self, goal):
        result = RefineAllResult()
        for target_object in goal.target_objects:
            refined_pose = self.__compute_refined_pose(target_object)
            result.refined_estimations.append(refined_pose)
        self._action_refine_all_server.set_succeeded(result)

    def __compute_refined_pose(self, target_object):
        initial_pose_estimate = target_object.pose.pose

        scene = self.__generate_scene(
            target_object.rgb_frame,
            target_object.depth_frame,
            initial_pose_estimate.position,
        )
        object3d = self.__generate_3d_object(target_object)

        ddope = dd.DiffDope(cfg=self.cfg, scene=scene, object3d=object3d)
        ddope.run_optimization()

        if (
            self.cfg.save_video
            and self.num_of_saved_videos_in_this_run < self.cfg.max_saved_videos_per_run
        ):
            self.__save_video(ddope)
            self.num_of_saved_videos_in_this_run += 1

            if (
                self.num_of_saved_videos_in_this_run
                == self.cfg.max_saved_videos_per_run
            ):
                rospy.loginfo(
                    f"No new videos will be saved in this run. This is because your config file dictates max of {self.cfg.max_saved_videos_per_run} saved videos per run."
                )
                rospy.loginfo(
                    'This is a mechanism to avoid saving too many videos during "continuous tracking" (refine_continuous.launch).'
                )
                rospy.loginfo(
                    "If you prefer, disable saving of video from the config file altogether."
                )

        refined_pose = self.__create_pose_stamped(ddope.get_pose())
        refined_target_object = TargetObject()
        refined_target_object.name = target_object.name
        refined_target_object.pose = refined_pose

        return refined_target_object

    def __save_video(self, ddope):
        rospy.loginfo("***You can disable video outputting from the config file***")
        video_dir = os.path.expanduser(self.cfg.saved_videos_path)
        self.__create_dir_if_not_exists(video_dir)
        next_video_id = self.__find_next_available_video_id(video_dir)
        video_path = os.path.join(video_dir, f"video_{next_video_id}.mp4")
        ddope.make_animation(output_file_path=video_path)
        rospy.loginfo(f"Video saved at {video_path}")

    def __convert_opengl_pose_to_opencv(self, transform):
        transform[0, 3] /= 10
        transform[1, 3] /= 10
        transform[2, 3] /= 10

        theta = np.pi
        rotation_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(theta), -np.sin(theta), 0],
                [0, np.sin(theta), np.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )

        return np.dot(rotation_matrix, transform)

    def __create_pose_stamped(self, diffdope_transform):
        diffdope_transform = self.__convert_opengl_pose_to_opencv(diffdope_transform)
        translation = diffdope_transform[0:3, 3]
        quaternion = tf_trans.quaternion_from_matrix(diffdope_transform)

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = "frame_id"

        pose_stamped.pose.position.x = translation[0]
        pose_stamped.pose.position.y = translation[1]
        pose_stamped.pose.position.z = translation[2]

        pose_stamped.pose.orientation.x = quaternion[0]
        pose_stamped.pose.orientation.y = quaternion[1]
        pose_stamped.pose.orientation.z = quaternion[2]
        pose_stamped.pose.orientation.w = quaternion[3]

        return pose_stamped

    def __create_dir_if_not_exists(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def __find_next_available_video_id(self, directory):
        max_id = 0
        pattern = r"video_(\d+)\.mp4"

        for filename in os.listdir(directory):
            if filename.endswith(".mp4"):
                match = re.search(pattern, filename)
                if match:
                    file_id = int(match.group(1))
                    max_id = max(max_id, file_id)

        return max_id + 1

    def __resize_image(self, image, is_depth=False):
        if self.cfg.image_resize < 1.0:
            if is_depth:
                image = cv2.resize(
                    image,
                    (
                        int(image.shape[1] * self.cfg.image_resize),
                        int(image.shape[0] * self.cfg.image_resize),
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                image = cv2.resize(
                    image,
                    (
                        int(image.shape[1] * self.cfg.image_resize),
                        int(image.shape[0] * self.cfg.image_resize),
                    ),
                )

        return image

    def __convert_rgb_frame_to_diffdope_image(self, cv_bridge, rgb_frame):
        try:
            cv_image = cv_bridge.imgmsg_to_cv2(rgb_frame, "bgr8")
            rgb_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            rgb_frame_normalized = rgb_frame / 255.0
            rgb_frame = cv2.flip(rgb_frame_normalized, 0)  # needed by diff-dope
        except Exception as e:
            rospy.logerr(f"CvBridge Error: {e}")

        rgb_frame = self.__resize_image(rgb_frame)
        rgb_tensor = torch.tensor(rgb_frame).float()
        rgb_image = dd.Image(img_tensor=rgb_tensor, img_resize=self.cfg.image_resize)
        return rgb_image

    def __convert_depth_frame_to_diffdope_image(self, cv_bridge, depth_frame):
        depth_frame = cv_bridge.imgmsg_to_cv2(depth_frame, depth_frame.encoding)
        depth_frame = depth_frame.astype(np.int16)
        depth_frame = cv2.flip(depth_frame, 0)  # needed by diff-dope
        depth_frame = self.__resize_image(depth_frame, is_depth=True)
        depth_tensor = torch.tensor(depth_frame).float()
        depth_image = dd.Image(
            img_tensor=depth_tensor, img_resize=self.cfg.image_resize, depth=True
        )

        return depth_image

    def __convert_segmentation_frame_to_diffdope_image(
        self, cv_bridge, rgb_frame, point_in_camera
    ):
        try:
            cv_image = cv_bridge.imgmsg_to_cv2(rgb_frame, "bgr8")
            rgb_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            segmentation_frame = self.segment_anything.segment(
                rgb_frame, point_in_camera
            )
        except Exception as e:
            rospy.logerr(f"CvBridge Error: {e}")

        # Make segmentation a 3 channel RGB image.
        segmentation_frame = cv2.merge(
            (segmentation_frame, segmentation_frame, segmentation_frame)
        )
        segmentation_frame = segmentation_frame / 255.0
        segmentation_frame = cv2.flip(segmentation_frame, 0)  # needed by diff-dope

        segmentation_frame = self.__resize_image(segmentation_frame)
        segmentation_tensor = torch.tensor(segmentation_frame).float()

        segmentation_image = dd.Image(
            img_tensor=segmentation_tensor, img_resize=self.cfg.image_resize
        )

        return segmentation_image

    def __generate_scene(self, rgb_frame, depth_frame, point_in_camera):
        cv_bridge = CvBridge()
        rgb_image = self.__convert_rgb_frame_to_diffdope_image(cv_bridge, rgb_frame)
        depth_image = self.__convert_depth_frame_to_diffdope_image(
            cv_bridge, depth_frame
        )
        segmentation_image = self.__convert_segmentation_frame_to_diffdope_image(
            cv_bridge, rgb_frame, point_in_camera
        )

        scene = dd.Scene(
            tensor_rgb=rgb_image,
            tensor_depth=depth_image,
            tensor_segmentation=segmentation_image,
            image_resize=self.cfg.image_resize,
        )

        return scene

    def __generate_3d_object(self, goal: TargetObject):
        dope_pose = goal.pose.pose
        dope_position = dope_pose.position
        position = np.array([dope_position.x, dope_position.y, dope_position.z])

        # Convert to mm
        position *= 1_000

        quat = dope_pose.orientation
        rotation = [quat.x, quat.y, quat.z, quat.w]

        model_path = os.path.join(diffdope_ros_path, goal.model_path)
        object3d = dd.Object3D(
            position, rotation, model_path=model_path, scale=goal.scale
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
