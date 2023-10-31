[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

# Diff-DOPE

![Diff-DOPE compared to megapose](figures/teaser.png)

A 6-DoF pose refiner that takes as input an image, a 3d model, and an initial object pose. The method then outputs the pose of the object using a differentiable renderer that minimizes the object reprojection error (rgb, depth, edges, mask, etc.).

Code and data to be release soon (pending internal approval).

## Installation
```bash
conda create -n diffdope python=3.9
conda activate diffdope
pip install -e .
```

## Run simple scene
From the root folder of the this repository, call:
```bash
python examples/simple_scene.py
```
The first run will compile several CUDA kernels, which will take a brief amount of time. Subsequent runs will be faster. After the script finishes, you should see the resulting object pose displayed as a matrix, as well as a filename for a video animation of the optimization process.

## Run on BOP
You can check `examples/run_bop_scene.py` to see how to load poses from a dataset and run optization. This uses the pertubed poses from the diff-dope paper. The paths in the file are absolute so you are going to need to change them. 

## Features and TODO
- Add a point cloud visualizer to check the output pose, use open 3d
- Add an example that uses a 3rd party neural network to add as a loss, canny detection, latent space
- The `find_crop` behaviour is not amazing, need to rework. 

## Development notes

Pre-commit hooks are available to ensure code quality. To install them, run:

```bash
pip install pre-commit
pre-commit install
```

To manually run the hooks without making a commit, call:

```bash
pre-commit run --all-files
```
