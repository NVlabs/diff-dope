import hydra
from omegaconf import DictConfig, OmegaConf

import diffdope as dd


@hydra.main(version_base=None, config_path="../configs/", config_name="diffdope")
def main(cfg: DictConfig):

    ddope = dd.DiffDope(cfg=cfg)

    # ddope.pose.reset_pose()
    ddope.set_batchsize(16)
    # ddope.object3d.mesh.enable_gradients_texture()
    print(ddope)
    # img_dd = dd.Image(img_path="data/example/scene/rgb.png", img_resize=0.5)
    # print(img_dd.img_tensor.shape)

    # img_dd = dd.Image(img_path="data/example/scene/seg.png", img_resize=0.5)
    # print(img_dd.img_tensor.shape)

    # run the network
    p = ddope.run_optimization()

    # cam = dd.Camera(100, 100, 50, 50, 100, 100)
    # ddope.camera = cam


# @hydra.main(version_base=None, config_path="../configs/", config_name="diffdope")
# def main(cfg: DictConfig):
#     ddope = dd.DiffDope(cfg=cfg)
#     ddope.pose.set_batchsize(16)

#     optimized_pose = ddope.optimize()
#     image = dd.render_pose(ddope, overlay=True, alpha=0.5, path="image1.png")


if __name__ == "__main__":
    main()
