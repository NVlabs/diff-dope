from setuptools import find_packages, setup

setup(
    name="diffdope",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "hydra-core",
        "numpy",
        "nvdiffrast@git+https://github.com/NVlabs/nvdiffrast.git",
        "omegaconf",
        "pyrr",
        "torch",
        "trimesh",
    ],
    extras_require={
        "dev": [
            "black",
            "ipython",
            "pre-commit",
            "pylint",
            "pytest",
        ],
    },
)
