import diffdope as dd

from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None,config_path="../configs/",config_name='diffdope')
def main(cfg: DictConfig):
	d_dope = dd.DiffDope(cfg = cfg)

	cam = dd.Camera(100,100,50,50,100,100)
	pose = dd.Pose([1,1,1],[0,0,0,1])

	hyperparameters = dd.Params(**cfg.parameters)

	mesh = dd.Mesh('path')

	images = dd.Images(path_rgb = '',path_depth = '',path_segmentation = '')

	d_dope.camera = cam 
	d_dope.pose = pose 
	d_dope.mesh = mesh
	d_dope.images = images
	d_dope.hyperparameters = hyperparameters

	d_dope.to_cuda()

	pose = d_dope.optimize()

	image = dd.render_pose(d_dope,overlay = True,alpha = 10,path='image1.png')
	dd.render_video_optimization(d_dope,path='my_vide0.mp4',fps=24)
	
	d_dope.hyperparameters.batchsize = 64
	d_dope.hyperparameters.learning_rates = [0.1,10]
	d_dope.hyperparameters.weights['rgb'] = 100
	d_dope.hyperparameters.weights['depth'] = 1
	d_dope.hyperparameters.weights['mask'] = 10
	d_dope.hyperparameters.use_canny = False

	pose = d_dope.optimize()


	d_dope.pose = Pose([10,10,10],[1,0,0,0])
	pose = d_dope.optimize()

	# came from a different place than a file 
	t = torch.zeros(3).cuda().grad()
	q = torch.zeros(4).cuda().grad()

	d_dope.pose = Pose(t,q)
	pose = d_dope.optimize()


if __name__ == "__main__":
    main()