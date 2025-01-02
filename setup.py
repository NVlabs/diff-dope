from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
    ext_modules=[
        CUDAExtension(
            name="renderutils_plugin",
            sources=[
                "src/diffdope/c_src/mesh.cu", 
                "src/diffdope/c_src/common.cpp", 
                "src/diffdope/c_src/torch_bindings.cpp"
            ],
            include_dirs=["src/diffdope/c_src/"],
            extra_compile_args={
                "cxx": ["-O3", "-DNVDR_TORCH"],
                "nvcc": ["-O3"]
            },
            extra_link_args=["-lcuda", "-lnvrtc"]
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
