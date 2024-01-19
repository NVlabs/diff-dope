import os
import sys
import rospkg
import hydra
import rospy
import torch
from icecream import ic
from omegaconf import DictConfig
import actionlib
from diffdope_ros.msg import RefineAllAction, RefineAllActionResult
from diffdope_ros.msg import RefineObjectAction, RefineObjectActionResult
from diffdope_ros.msg import ObjectDetails
from diffdope_ros.msg import TargetObject
import diffdope as dd


class DiffDOPEServer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self._action_refine_object_server = actionlib.SimpleActionServer(
            'refine_object', 
            RefineObjectAction, 
            execute_cb=self.refine_object, 
            auto_start=False)

        self._action_refine_all_server = actionlib.SimpleActionServer(
            'refine_all', 
            RefineAllAction, 
            execute_cb=self.refine_all, 
            auto_start=False)

        self._action_refine_object_server.start()
        self._action_refine_all_server.start()
        rospy.loginfo('[DiffDOPE Server] Both action servers started. Waiting for requests...')

    def refine_object(self, target_object: TargetObject):
        result = self._compute_refined_pose(target_object, RefineObjectActionResult())
        self._action_refine_object_server.set_succeeded(result)

    def refine_all(self, target_objects):
        for target_object in target_objects:
            result = self._compute_refined_pose(target_object, RefineObjectActionResult())
            self._action_refine_all_server.set_succeeded(result)

    def _create_pose_stamped(self, diffdope_pose):
        return PoseStamped()

    def _compute_refined_pose(self, target_object: TargetObject, result):
        scene = self._generate_scene(target_object.rgb_frame, target_object.depth_frame)
        object3d = self._generate_3d_object(target_object)

        ddope = dd.DiffDope(cfg=self.cfg, scene=scene, object3d=object3d)
        ddope.run_optimization()

        refined_pose = self._create_pose_stamped(ddope.get_pose())
        target_object = TargetObject()
        target_object.name = target_object.name
        target_object.pose = refined_pose
        result.refined_estimation = target_object

        return result

    def _generate_scene(self, rgb_frame, depth_frame):
        rgb_tensor = torch.tensor(rgb_frame).float()
        rgb_image = dd.Image(img_tensor=rgb_tensor)

        depth_tensor = torch.tensor(depth_frame).float()
        depth_image = dd.Image(img_tensor=depth_tensor, depth=True)

        scene = dd.Scene(tensor_rgb=rgb_image, tensor_depth=depth_image)
        return scene

    def _generate_3d_object(self, goal: ObjectDetails):
        pos = goal.pose.position
        position = [pos.x, pos.y, pos.z]

        quat = goal.pose.orientation
        rotation = [quat.x, quat.y, quat.z, quat.w]  # respecting diffdope order

        object3d = dd.Object3D(
            position, rotation, model_path=goal.model_path, scale=goal.scale
        )

        return object3d


def parse_cfg():
    diffdope_ros_path = rospkg.RosPack().get_path('diffdope_ros')
    config_dir = os.path.join(diffdope_ros_path, "configs")
    config_file = sys.argv[1]

    hydra.initialize_config_dir(version_base=None, config_dir=config_dir)
    cfg = hydra.compose(config_name=config_file, return_hydra_config=True)

    return cfg


if __name__ == "__main__":
    rospy.init_node('diffdope_action_server')

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        rospy.loginfo(f"[DiffDOPE Server] Config file working with: {config_file}")
    else:
        sys.exit("No config file provided.")

    cfg = parse_cfg()
    server = DiffDOPEServer(cfg)
    rospy.spin()
