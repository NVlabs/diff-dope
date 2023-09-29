[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

# Diff-DOPE

![Diff-DOPE compared to megapose](figures/teaser.png)

A 6-DoF pose refiner that takes as input an image, a 3d model, and an initial object pose. The method then outputs the pose of the object using a differentiable renderer that minimizes the object reprojection error (rgb, depth, edges, mask, etc.).

Code and data to be release soon (pending internal approval).

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
