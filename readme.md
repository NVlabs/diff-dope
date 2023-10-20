[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

# Diff-DOPE

![Diff-DOPE compared to megapose](figures/teaser.png)

A 6-DoF pose refiner that takes as input an image, a 3d model, and an initial object pose. The method then outputs the pose of the object using a differentiable renderer that minimizes the object reprojection error (rgb, depth, edges, mask, etc.).

Code and data to be release soon (pending internal approval).

## Installation
```bash
pip install -r requirements.txt
pip install -e . 
```

## Running simple scene
From the root folder of the this repo call. 
```bash
python examples/simple_scene.py
```
You should see an object pose express as a matrix displayed. 


## Features and TODO
- Add a point cloud visualizer to check the output pose, use open 3d 
- Add an example that runs on the data format bop
- Add an example that uses a 3rd party neural network to add as a loss, canny detection, latent space 

## Development notes

Pre-commit hooks are available to ensure code quality. To install them, run:

```bash
pip install pre-commit
pre-commit install
```

To manually run the hooks without making a commit, call:

```bash
pre-commit run --all-files
``
