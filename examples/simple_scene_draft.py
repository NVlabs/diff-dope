import diffdope as dd

from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None,config_path="../configs/",config_name='diffdope')
def main(cfg: DictConfig):

	# create diff-dope to optimize
	d_dope = dd.DiffDope(cfg = cfg)

	# a new camera
	cam = dd.Camera(100,100,50,50,100,100)

	# a new pose
	pose = dd.Pose([1,1,1],[0,0,0,1])

	# load the hyper params from the config file
	hyperparameters = dd.Params(**cfg.parameters)

	# load a mesh data
	mesh = dd.Mesh('path')

	# load the data for a scene to optimize
	images = dd.Images(path_rgb = '',path_depth = '',path_segmentation = '')

	# update diff dope to have all this new data. 
	d_dope.camera = cam 
	d_dope.pose = pose 
	d_dope.mesh = mesh
	d_dope.images = images
	d_dope.hyperparameters = hyperparameters

	# put things on the gpu
	d_dope.to_cuda()

	# run the optimization and get the best pose
	pose = d_dope.optimize()

	# render an image from the optimization that was run
	image = dd.render_pose(d_dope,overlay = True,alpha = 10,path='image1.png')

	# render the optimization visualization for the the best pose
	dd.render_video_optimization(d_dope,path='my_vide0.mp4',fps=24)
	
	# chanse some hyperparams to play with the diff-dope
	d_dope.hyperparameters.batchsize = 64
	d_dope.hyperparameters.learning_rates = [0.1,10]
	d_dope.hyperparameters.weights['rgb'] = 100
	d_dope.hyperparameters.weights['depth'] = 1
	d_dope.hyperparameters.weights['mask'] = 10
	d_dope.hyperparameters.use_canny = False

	pose = d_dope.optimize()



	# came from a different place than a file 
	#  could be a diff pnp 
	pose_from_pnp = super_process_pnp()

	d_dope.pose = Pose(
		pose_from_pnp['position'],
		pose_from_pnp['rotation']
	)
	pose = d_dope.optimize()


if __name__ == "__main__":
    main()