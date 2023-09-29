import hydra
from omegaconf import DictConfig, OmegaConf

import diffdope as dd


@hydra.main(version_base=None, config_path="../configs/", config_name="diffdope")
def main(cfg: DictConfig):
    ddope = dd.DiffDope(cfg=cfg)

    ddope.pose.reset_pose()
    ddope.pose.set_batchsize(16)

    img_dd = dd.Image(img_path="data/example/scene/rgb.png", img_resize=0.5)
    print(img_dd)
    # run the network
    p = ddope.pose()

    cam = dd.Camera(100, 100, 50, 50, 100, 100)
    ddope.camera = cam


# @hydra.main(version_base=None, config_path="../configs/", config_name="diffdope")
# def main(cfg: DictConfig):
#     ddope = dd.DiffDope(cfg=cfg)
#     ddope.pose.set_batchsize(16)

#     optimized_pose = ddope.optimize()
#     image = dd.render_pose(ddope, overlay=True, alpha=0.5, path="image1.png")


if __name__ == "__main__":
    main()
