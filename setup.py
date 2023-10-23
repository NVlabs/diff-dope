from setuptools import find_packages, setup

setup(
    name="diffdope",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "hydra-core",
        "icecream",
        "imageio[ffmpeg]",
        "matplotlib",
        "ninja",
        "numpy",
        "nvdiffrast@git+https://github.com/NVlabs/nvdiffrast.git",
        "omegaconf",
        "opencv-python",
        "pillow",
        "pyglet==1.5.27",
        "pyrr",
        "tdqm",
        "torch",
        "trimesh[easy]==3.21.5",
    ],
    extras_require={
        "dev": [
            "pre-commit",
        ],
    },
)
