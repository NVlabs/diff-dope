import copy
import os
import threading
import time
from functools import partial

import cv2
import hydra
import rospy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from icecream import ic
from omegaconf import DictConfig, OmegaConf, open_dict
from sensor_msgs.msg import Image as ROS_Image

import diffdope as dd


def pose_callback(pose_stamped_msg, object_name):
    with pose_lock:
        live_dope_pose_per_object[object_name] = pose_stamped_msg.pose


def rgb_callback(img_msg, object_name):
    bridge = CvBridge()
    with rgb_lock:
        try:
            cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
            rgb_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            rgb_frame_normalized = rgb_frame / 255.0
            flipped_rgb_frame_normalized = cv2.flip(
                rgb_frame_normalized, 0
            )  # needed by diff-dope
            live_rgb_per_object[object_name] = flipped_rgb_frame_normalized
        except Exception as e:
            rospy.logerr(f"CvBridge Error: {e}")


def depth_callback(depth_msg, object_name):
    bridge = CvBridge()
    depth_scale = 100

    with depth_lock:
        try:
            cv_image_depth = (
                bridge.imgmsg_to_cv2(depth_msg, "passthrough") / depth_scale
            )
            depth_frame = cv2.flip(cv_image_depth, 0)  # needed by diff-dope
            live_depth_per_object[object_name] = depth_frame
        except Exception as e:
            rospy.logerr(f"CvBridge Error: {e}")


def init_ros_subscribers(cfg):
    rospy.init_node("diffdope", anonymous=True)

    for obj in cfg["objects"]:
        object_name = obj["name"]

        # Create pose subscriber (DOPE)
        live_dope_pose_per_object[object_name] = None
        pose_topic = obj.pose_topic
        pose_sub = rospy.Subscriber(
            pose_topic, PoseStamped, partial(pose_callback, object_name=object_name)
        )
        subscribers.append(pose_sub)

        # Create RGB frame subscriber
        live_rgb_per_object[object_name] = None
        rgb_topic = cfg.topics.rgb
        rgb_sub = rospy.Subscriber(
            rgb_topic, ROS_Image, partial(rgb_callback, object_name=object_name)
        )
        subscribers.append(rgb_sub)

        # Create depth frame subscriber
        live_depth_per_object[object_name] = None
        depth_topic = cfg.topics.depth
        depth_sub = rospy.Subscriber(
            depth_topic, ROS_Image, partial(depth_callback, object_name=object_name)
        )
        subscribers.append(depth_sub)

    # Allow time for subscribers to initialise first pose/frames.
    time.sleep(1)


def get_initialisation_for(object_name):
    dope_pose, rgb_frame, depth_frame = None, None, None

    # Prevent data race by using a lock
    with pose_lock, rgb_lock, depth_lock:
        dope_pose = live_dope_pose_per_object[object_name]
        rgb_frame = live_rgb_per_object[object_name]
        depth_frame = live_depth_per_object[object_name]

    return dope_pose, rgb_frame, depth_frame


def get_refined_pose_using_diff_dope(cfg, scene, object3d):
    ddope = dd.DiffDope(cfg=cfg, scene=scene, object3d=object3d)
    ddope.run_optimization()
    ic(ddope.get_argmin(), ddope.get_pose())
    ddope.make_animation(output_file_path="ros_scene.mp4")
    print("Saved animation to ros_scene.mp4")
    return ddope.get_pose()


def generate_scene(rgb_frame, depth_frame):
    rgb_tensor = torch.tensor(rgb_frame).float()
    rgb_image = dd.Image(img_tensor=rgb_tensor)

    depth_tensor = torch.tensor(depth_frame).float()
    depth_image = dd.Image(img_tensor=depth_tensor, depth=True)

    scene = dd.Scene(
        tensor_rgb=rgb_image,
        tensor_depth=depth_image,
        path_segmentation="data/example/segmentation.png",
        image_resize=1.0,
    )
    return scene


def generate_3d_object(obj_cfg, init_pose):
    pos = init_pose.position
    position = [pos.x, pos.y, pos.z]

    quat = init_pose.orientation
    rotation = [quat.x, quat.y, quat.z, quat.w]  # respecting diffdope order

    object3d = dd.Object3D(
        position, rotation, model_path=obj_cfg["model_path"], scale=obj_cfg["scale"]
    )
    return object3d


@hydra.main(
    version_base=None, config_path="../configs/", config_name="multiobject_with_dope"
)
def main(cfg: DictConfig):
    init_ros_subscribers(cfg)

    for obj_cfg in cfg["objects"]:
        init_pose, rgb_frame, depth_frame = get_initialisation_for(obj_cfg.name)

        scene = generate_scene(rgb_frame, depth_frame)
        object3d = generate_3d_object(obj_cfg, init_pose)

        refined_pose = get_refined_pose_using_diff_dope(cfg, scene, object3d)
        print(f"The initial pose for {obj_cfg.name} was {init_pose}")
        print(f"The refined pose for {obj_cfg.name} is {refined_pose}")


if __name__ == "__main__":
    subscribers = []
    pose_lock = threading.Lock()
    live_dope_pose_per_object = {}

    rgb_lock = threading.Lock()
    live_rgb_per_object = {}

    depth_lock = threading.Lock()
    live_depth_per_object = {}

    main()
