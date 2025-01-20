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
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from omegaconf import DictConfig
from sensor_msgs.msg import Image as ROS_Image

from diffdope_ros.msg import (
    RefineAllAction,
    RefineAllGoal,
    RefineObjectAction,
    RefineObjectGoal,
    TargetObject,
)


def pose_callback(pose_stamped_msg, object_name):
    global live_dope_pose_per_object
    with pose_lock:
        live_dope_pose_per_object[object_name] = pose_stamped_msg


def rgb_callback(img_msg):
    global live_rgb
    with rgb_lock:
        live_rgb = img_msg


def depth_callback(depth_msg):
    global live_depth
    with depth_lock:
        live_depth = depth_msg


def init_ros_subscribers(cfg):
    rospy.init_node("diffdope", anonymous=True)

    # Create RGB frame subscriber
    rgb_topic = cfg.topics.rgb
    rgb_sub = rospy.Subscriber(rgb_topic, ROS_Image, rgb_callback)
    subscribers.append(rgb_sub)

    # Create depth frame subscriber
    depth_topic = cfg.topics.depth
    depth_sub = rospy.Subscriber(depth_topic, ROS_Image, depth_callback)
    subscribers.append(depth_sub)

    for obj in cfg["objects"]:
        object_name = obj["name"]

        # Create pose subscriber (DOPE)
        live_dope_pose_per_object[object_name] = None
        pose_topic = obj.pose_topic
        pose_sub = rospy.Subscriber(
            pose_topic, PoseStamped, partial(pose_callback, object_name=object_name)
        )
        subscribers.append(pose_sub)

    # Allow time for subscribers to initialise first pose/frames.
    time.sleep(1)


def get_initialisation_for(object_name):
    dope_pose, rgb_frame, depth_frame = None, None, None

    # Prevent data race by using a lock
    while dope_pose is None or rgb_frame is None or depth_frame is None:
        with pose_lock, rgb_lock, depth_lock:
            dope_pose = live_dope_pose_per_object[object_name]
            rgb_frame = live_rgb
            depth_frame = live_depth

    return dope_pose, rgb_frame, depth_frame


def get_object_definition_by_name(cfg, object_name):
    for obj_cfg in cfg.objects:
        if obj_cfg.name == object_name:
            return obj_cfg


def wait_until_all_data_available():
    rospy.loginfo("Waiting for all data to become available ...")

    while True:
        if live_rgb is None:
            rospy.loginfo("Waiting for rgb frames ...")
        elif live_depth is None:
            rospy.loginfo("Waiting for depth frames ...")
        elif any([v is None for _, v in live_dope_pose_per_object.items()]):
            rospy.loginfo("Waiting for some initial poses ...")
            for k, v in live_dope_pose_per_object.items():
                if v is None:
                    rospy.loginfo(f"The pose of object {k} isn't available yet ...")
        else:
            break
        time.sleep(0.5)

    rospy.loginfo("All data available. Starting refinement.")


def main_single_object(cfg: DictConfig, object_name):
    client = actionlib.SimpleActionClient("refine_object", RefineObjectAction)
    client.wait_for_server()
    obj_cfg = get_object_definition_by_name(cfg, object_name)
    publisher = rospy.Publisher(f"diffdope_{obj_cfg.name}", PoseStamped, queue_size=1)

    try:
        while True:
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
            result = client.get_result()
            publisher.publish(result.refined_estimation.pose)
    except KeyboardInterrupt:
        return


def main_all_objects(cfg: DictConfig):
    client = actionlib.SimpleActionClient("refine_all", RefineAllAction)
    client.wait_for_server()

    try:
        while True:
            goal = RefineAllGoal()
            for obj_cfg in cfg.objects:
                init_pose, rgb_frame, depth_frame = get_initialisation_for(obj_cfg.name)
                target_object = TargetObject()
                target_object.name = obj_cfg.name
                target_object.pose = init_pose
                target_object.rgb_frame = rgb_frame
                target_object.depth_frame = depth_frame
                target_object.model_path = obj_cfg.model_path
                target_object.scale = obj_cfg.scale
                goal.target_objects.append(target_object)

            client.send_goal(goal)
            client.wait_for_result()

            result = client.get_result()
            for refined_estimation in result.refined_estimations:
                object_name = refined_estimation.name
                pose = refined_estimation.pose
                publisher = rospy.Publisher(
                    f"diffdope_{object_name}", PoseStamped, queue_size=1
                )
                publisher.publish(pose)

    except KeyboardInterrupt:
        return


def parse_cfg():
    diffdope_ros_path = rospkg.RosPack().get_path("diffdope_ros")
    config_dir = os.path.join(diffdope_ros_path, "configs")
    config_file = sys.argv[1]

    hydra.initialize_config_dir(version_base=None, config_dir=config_dir)
    cfg = hydra.compose(config_name=config_file, return_hydra_config=True)

    return cfg


def is_object_name_in_cfg(cfg, object_name):
    for obj_cfg in cfg.objects:
        if obj_cfg.name == object_name:
            return True


if __name__ == "__main__":
    bridge = CvBridge()
    subscribers = []
    pose_lock = threading.Lock()
    live_dope_pose_per_object = {}

    rgb_lock = threading.Lock()
    live_rgb = None

    depth_lock = threading.Lock()
    live_depth = None

    cfg = parse_cfg()
    init_ros_subscribers(cfg)
    wait_until_all_data_available()

    if cfg.save_video:
        rospy.loginfo(
            "Note you have save_video to True in config file; you may want to turn it off."
        )

    object_name = sys.argv[2]
    if object_name == "all":
        rospy.loginfo("Refining all objects...")
        main_all_objects(cfg)
    else:
        rospy.loginfo(f"Refining {object_name} only...")
        if not is_object_name_in_cfg(cfg, object_name):
            sys.exit("The object name provided is not in the config file.")
        main_single_object(cfg, object_name)
