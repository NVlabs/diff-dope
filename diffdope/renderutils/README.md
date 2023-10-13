# renderutils

## Getting started

Requires Python 3.6+, VS2019+, Cuda 11.3+ and PyTorch 1.10+

Tested in Anaconda3 with Python 3.10 and PyTorch 2.0

## One time setup

Install the [Cuda toolkit](https://developer.nvidia.com/cuda-toolkit) (required to build the PyTorch extensions).
We support Cuda 11.3 and above.
Pick the appropriate version of PyTorch compatible with the installed Cuda toolkit.
Below is an example with Cuda 11.7

```
conda create -n dmodel python=3.10
activate dmodel
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install ninja
```

## Validate correctness 

```
activate dmodel
cd tests
python test_mesh.py  
```

## Performance benchmark   

```
activate dmodel
cd tests
python perf_mesh.py  
```






