import os
import sys
import threading
import time
from functools import partial

import actionlib
import cv2
import hydra
import rospkg
import rospy
from diffdope_ros.msg import RefineObjectAction, RefineObjectGoal, TargetObject
from geometry_msgs.msg import PoseStamped
from omegaconf import DictConfig
from sensor_msgs.msg import Image as ROS_Image


def pose_callback(pose_stamped_msg, object_name):
    with pose_lock:
        live_dope_pose_per_object[object_name] = pose_stamped_msg


def rgb_callback(img_msg, object_name):
    with rgb_lock:
        live_rgb_per_object[object_name] = img_msg
    # bridge = CvBridge()
    # with rgb_lock:
    #     try:
    #         cv_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    #         rgb_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    #         rgb_frame_normalized = rgb_frame / 255.0
    #         flipped_rgb_frame_normalized = cv2.flip(
    #             rgb_frame_normalized, 0
    #         )  # needed by diff-dope
    #         live_rgb_per_object[object_name] = flipped_rgb_frame_normalized
    #     except Exception as e:
    #         rospy.logerr(f"CvBridge Error: {e}")


def depth_callback(depth_msg, object_name):
    with depth_lock:
        live_depth_per_object[object_name] = depth_msg

    # bridge = CvBridge()
    # depth_scale = 100

    # with depth_lock:
    #     try:
    #         cv_image_depth = (
    #             bridge.imgmsg_to_cv2(depth_msg, "passthrough") / depth_scale
    #         )
    #         depth_frame = cv2.flip(cv_image_depth, 0)  # needed by diff-dope
    #         live_depth_per_object[object_name] = depth_frame
    #     except Exception as e:
    #         rospy.logerr(f"CvBridge Error: {e}")


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


def main_all_objects(cfg: DictConfig):
    for obj_cfg in cfg.objects:
        init_pose, rgb_frame, depth_frame = get_initialisation_for(obj_cfg.name)
        refined_pose = get_refined_pose_using_diff_dope(cfg, scene, object3d)
        print(f"The initial pose for {obj_cfg.name} was {init_pose}")
        print(f"The refined pose for {obj_cfg.name} is {refined_pose}")


def get_object_definition_by_name(cfg, object_name):
    for obj_cfg in cfg.objects:
        if obj_cfg.name == object_name:
            return obj_cfg


def main_single_object(cfg: DictConfig, object_name):
    client = actionlib.SimpleActionClient("refine_object", RefineObjectAction)
    client.wait_for_server()
    obj_cfg = get_object_definition_by_name(cfg, object_name)

    init_pose, rgb_frame, depth_frame = get_initialisation_for(obj_cfg.name)

    goal = RefineObjectGoal()
    target_object = TargetObject()
    target_object.name = obj_cfg.name
    target_object.pose = init_pose
    target_object.rgb_frame = rgb_frame
    target_object.depth_frame = depth_frame
    target_object.model_path = obj_cfg.model_path
    target_object.scale = obj_cfg.scale
    goal.target_object = target_object

    client.send_goal(goal)
    client.wait_for_result()

    refined_pose = client.get_result()
    print(refined_pose)


def parse_cfg():
    diffdope_ros_path = rospkg.RosPack().get_path("diffdope_ros")
    config_dir = os.path.join(diffdope_ros_path, "configs")
    config_file = sys.argv[1]

    hydra.initialize_config_dir(version_base=None, config_dir=config_dir)
    cfg = hydra.compose(config_name=config_file, return_hydra_config=True)

    return cfg


if __name__ == "__main__":
    subscribers = []
    pose_lock = threading.Lock()
    live_dope_pose_per_object = {}

    rgb_lock = threading.Lock()
    live_rgb_per_object = {}

    depth_lock = threading.Lock()
    live_depth_per_object = {}

    cfg = parse_cfg()
    init_ros_subscribers(cfg)
    main_single_object(cfg, "bbq_sauce")
