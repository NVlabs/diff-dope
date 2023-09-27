import diffdope as dd

from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None,config_path="../configs/",config_name='diffdope')
def main(cfg: DictConfig):
	ddope = dd.DiffDope(cfg = cfg)
	
	ddope.pose.reset_pose()
	ddope.pose.set_batchsize(16)

	p = ddope.pose()	
	
	cam = dd.Camera(100,100,50,50,100,100)
	ddope.camera = cam

if __name__ == "__main__":
    main()