import os

import cv2
import json
import hydra
from icecream import ic
from omegaconf import DictConfig, OmegaConf

import diffdope as dd


@hydra.main(version_base=None, config_path="../configs/", config_name="diffdope")
def main(cfg: DictConfig):

    # basic ddope object
    ddope = dd.DiffDope(cfg=cfg)

    # load a scene 
    path_scene_bop = '/home/jtremblay/code/camera2robot/hope/val/000001/'

    # models path
    path_scene_bop_models = '/home/jtremblay/code/camera2robot/hope/models/'

    # scene error generated for the paper. 
    path_error_scene_bop = '/home/jtremblay/code/diff-dope/data/hope/val/000001/scene_error_deg_040_trans_016.json'

    with open(f"{path_error_scene_bop}", 'r') as f:
        data_scene = json.load(f)

    # load frame 0 
    frame = "0" 
    frame_0 = data_scene[frame]

    # keep track of the models loaded
    loaded_models = {}

    # load the depth and rgb image 
    scene = dd.Scene(
        path_img = f"{path_scene_bop}/rgb/{frame.zfill(6)}.png",
        path_depth = f"{path_scene_bop}/depth/{frame.zfill(6)}.png",
        path_segmentation = f"{path_scene_bop}/rgb/{frame.zfill(6)}.png",
        image_resize = cfg.scene.image_resize
    )
    scene.cuda()
    scene.set_batchsize(cfg.hyperparameters.batchsize)


    for i_obj, obj in enumerate(frame_0):

        # load the object
        if not obj['obj_id'] in loaded_models:
            loaded_models[obj['obj_id']] = dd.Mesh(f'{path_scene_bop_models}/obj_{str(obj["obj_id"]).zfill(6)}.ply',
                scale = 0.01
            )
            loaded_models[obj['obj_id']].set_batchsize(cfg.hyperparameters.batchsize)
            loaded_models[obj['obj_id']].cuda()

            # ic(loaded_models[obj['obj_id']])


        # load the pose
        pose_to_update = dd.Object3D(
            position = obj['cam_t_m2c'],
            rotation = obj['cam_R_m2c'],
            scale = 0.01,
            batchsize = cfg.hyperparameters.batchsize
        )
        pose_to_update.mesh = loaded_models[obj['obj_id']]
        pose_to_update.cuda()

        # load the segmentation
        mask = dd.Image(
            img_path = f"{path_scene_bop}/mask_visib/{frame.zfill(6)}_{str(i_obj).zfill(6)}.png",
            img_resize = cfg.scene.image_resize
        )
        ic(mask)
        mask.cuda()
        mask.set_batchsize(cfg.hyperparameters.batchsize)
        
        # set things for the optimization        
        scene.tensor_segmentation = mask
        ddope.scene = scene
        ddope.object3d = pose_to_update

        # run the optimiztion
        ddope.run_optimization()
        
        # Output pose
        ic(f'object {i_obj}',ddope.get_argmin(), ddope.get_pose())

        # render an image
        img = ddope.render_img()
        cv2.imwrite(f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/{str(i_obj).zfill(2)}.png", img)
        
        # ddope.make_animation(output_file_path=
        #     f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/{str(i_obj).zfill(2)}.mp4")

if __name__ == "__main__":
    main()
