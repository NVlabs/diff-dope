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
        # The following might be requirements for nvdiffrast
        # (they are installed by pip in the Dockerfile),
        # but we should check if they are really needed:
        # "ninja",
        # "imageio",
        # "imageio-ffmpeg",
    ],
    extras_require={
        "dev": [
            "black",
            "ipython",
            "pre-commit",
        ],
    },
)
