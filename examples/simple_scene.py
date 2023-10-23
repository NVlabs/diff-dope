import os

import cv2
import hydra
from icecream import ic
from omegaconf import DictConfig, OmegaConf

import diffdope as dd


@hydra.main(version_base=None, config_path="../configs/", config_name="diffdope")
def main(cfg: DictConfig):
    # load the optimization through the diffdope config file
    ddope = dd.DiffDope(cfg=cfg)

    # run the optimization
    ddope.run_optimization()

    ic(ddope.get_argmin(), ddope.get_pose())

    # get the loss plot for the argmin of the optimization
    img_plot = ddope.plot_losses()
    cv2.imwrite("plot.png", img_plot)

    # save the video of optimization animation
    ddope.make_animation(output_file_path="simple_scene.mp4")
    print("Saved animation to simple_scene.mp4")


if __name__ == "__main__":
    main()
