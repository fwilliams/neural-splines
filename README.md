# Neural Splines: Fitting 3D Surfaces with Inifinitely-Wide Neural Networks
![Neural Splines Teaser](https://github.com/fwilliams/neural-splines/blob/master/imgs/teaser.png)
This repository contains the official implementation of the CVPR 2021 (Oral) paper [Neural Splines: Fitting 3D Surfaces with Infinitely-Wide Neural Networks](https://arxiv.org/abs/2006.13782).

## Table of Contents
- [Setup Instructions](#setup-instructions)
  * [System Requirements](#system-requirements)
  * [Installing Dependencies](#installing-dependencies)
    + [Installing Dependencies with `conda`](#installing-dependencies-with--conda-)
    + [Installing Dependencies with `pip`](#installing-dependencies-with--pip-)
    + [Installing Dependencies Manually (Not Recommended)](#installing-dependencies-manually--not-recommended-)
  * [Testing Your Installation](#testing-your-installation)
- [Using Neural Splines from the Command Line](#using-neural-splines-from-the-command-line)
    + [Reconstructing a point cloud with `fit.py`](#reconstructing-a-point-cloud-with--fitpy-)
    + [Reconstructing very large point clouds with `fit-grid.py`](#reconstructing-very-large-point-clouds-with--fit-gridpy-)
    + [Additional arguments to `fit.py` and `fit-grid.py`](#additional-arguments-to--fitpy--and--fit-gridpy-)
- [Using Neural Splines in Python](#using-neural-splines-in-python)

## Setup Instructions

### System Requirements
Neural Splines uses [FALKON](https://arxiv.org/abs/1705.10958), a state-of-the-art kernel ridge regression solver to fit 
surfaces on one or more GPUs. We thus require at least one GPU to run Neural Splines. We additionally require a working version of the CUDA compiler `nvcc`.
We recommend running this code on a machine with a lot of memory if you want to reconstruct large point clouds 
since Neural Splines stores an MxM preconditioner matrix in CPU memory (where M is the number of Nystrom samples). 

### Installing Dependencies
Neural splines has several dependencies which must be installed before it can be used. Some of these dependencies must be built and take time to install. 
There are three ways to install dependencies:

#### Installing Dependencies with `conda`
Simply run
```
conda env create -f environment.yml
```
and then go grab a coffee ☕. When you get back, you will have a conda environment called `neural-splines` with the right dependencies installed.

#### Installing Dependencies with `pip`
We include several `requirement-*.txt` files in the `requirements` directory depending on your version of cuda. Choose the right file for your installation then run
```
pip install -r requirements/requirements-cuda<VERSION>.txt
```
and then go grab a coffee ☕.

#### Installing Dependencies Manually (Not Recommended)
You will need to install the following dependencies manually to use this repository:
* [PyTorch](https://pytorch.org/)
* [scikit-image](https://scikit-image.org/)
* [numpy](https://numpy.org/)

You will also need to build the following dependencies from source. The easiest way to do this is with `pip` (see commands below), but you can also clone the linked repositories and run `setup.py install`:
* [point-cloud-utils](https://github.com/fwilliams/point-cloud-utils/tree/neural-splines): `pip install git+https://github.com/fwilliams/point-cloud-utils.git@neural-splines`
* [FALKON](https://github.com/fwilliams/falkon/tree/kml): `pip install git+https://github.com/fwilliams/falkon.git@kml`
* [KeOps](https://github.com/fwilliams/keops/tree/falkon)`pip install git+https://github.com/fwilliams/keops.git@falkon`

### Testing Your Installation
⚠️ **WARNING** ⚠️ Due to a bug in [KeOps](https://www.kernel-operations.io/keops/index.html), the first time you use any code in this repository will throw a `ModuleNotFoundError`. All subsequent invocations of Neural Splines should work.

1. Download and unzip the [example point clouds here](http://storage.googleapis.com/local-implicit-grids/demo_data.zip)
2. Unzip the file, in the directory of this repository, which should produe a directory named `demo_data`
3. Run `python fit.py demo_data/bunny.ply 0.005 10_000 128` On the first run this will fail (see above, just rerun it). On the second run it will compile some kernels and then produce a file called `recon.ply` which should be a reconstructed Stanford Bunny. The image below shows the input points and reconstruction for the bunny,
5. Run `python fit-grid.py demo_data/living_room_33_500_per_m2.ply 0.005 10_000 512 8` which will produce another `recon.ply` mesh, this time of a full room as shown below.
<p align="center">
    <span>
        <img src="https://github.com/fwilliams/neural-splines/blob/master/imgs/bunny.png" alt="A reconstructed Stanford Bunny" width="33%">
        <img src="https://github.com/fwilliams/neural-splines/blob/master/imgs/room.png" alt="A reconstruced living room" width="33%">
    </span>
</p>


## Using Neural Splines from the Command Line
There are two scripts in this repository to fit surfaces from the command line:
* `fit.py` fits an input point cloud using a single Neural Spline. This method is good for smaller inputs without too much geometric complexity.
* `fit_grid.py` fits an input point cloud in chunks using a different Neural Spline per chunk. This method is better for very large scenes with a lot of geometric complexity.

#### Reconstructing a point cloud with `fit.py`

`fit.py` fits an input point cloud using a single Neural Spline. This approach works best for relatively small inputs which don't have too much geometric complexity. `fit.py` takes least the following arguments
```
usage: fit.py <INPUT_POINT_CLOUD> <EPS> <NUM_NYSTROM_SAMPLES> <GRID_SIZE>
```
where
* **`<INPUT_POINT_CLOUD>`** is a path to a PLY file containing 3D points and corresponding normals
* **`<EPS>`** is a spacing parameter used for finite difference approximation of the gradient term in the kernel. 
  To capture all surface details this should be less than half the smallest distance between two points. 
  Generally setting this to values smalelr than `0.5/grid_size` is reasonable for this parameter
* **`<NUM_NYSTROM_SAMPLES>`** is the number of points to use as basis centers. A larger number of Nystrom samples will yield 
  a more accurate reconstruction but increase runtime and CPU memory usage. Generally good values for this are between 
  `10*sqrt(N)` and `100*sqrt(N)` where `N` is the number of input points.
* **`<GRID_SIZE>`** is the number of voxel cells along the longest axis of the bounding box on which the reconstructed 
  function gets sampled. For example if `<grid_size>` is `128` and the bounding box of the input pointcloud has dimensions `[1, 0.5, 0.5]`, then we will sample the function on a `128x64x64` voxel grid before extracting a mesh.
 
 

#### Reconstructing very large point clouds with `fit-grid.py`

`fit-grid.py` fits an input point cloud in chunks using a different Neural Spline per chunk. This approach works well when the input point cloud is large or has a lot of geometric complexity. `fit-grid.py` takes the following arguments
```
usage: fit-grid.py <INPUT_POINT_CLOUD> <EPS> <NUM_NYSTROM_SAMPLES> <GRID_SIZE> <CELLS_PER_AXIS> --overlap <OVERLAP>
```
where
* **`<INPUT_POINT_CLOUD>`** is a path to a PLY file containing 3D points and corresponding normals
* **`<EPS>`** is a spacing parameter used for finite difference approximation of the gradient term in the kernel. 
  To capture all surface details this should be less than half the smallest distance between two points. 
  Generally setting this to values smaller than `0.5/grid_size` is reasonable for this parameter
* **`<NUM_NYSTROM_SAMPLES>`** is the number of points to use as basis centers within each chunk. A larger number of Nystrom samples will yield 
  a more accurate reconstruction but increase runtime and CPU memory usage.
* **`<GRID_SIZE>`** is the number of voxel cells along the longest axis of the bounding box on which the reconstructed 
  function gets sampled. For example if `<GRID_SIZE>` is `128` and the bounding box of the input pointcloud has dimensions `[1, 0.5, 0.5]`, then we will sample the function on a `128x64x64` voxel grid before extracting a mesh.
* **`<CELLS_PER_AXIS>`** is an integer specifying the number of chunks to use along each axis. E.g. if `<cells-per-axis>` is 8, we will reconstruct the surface using 8x8x8 chunks.
* **`--overlap <OVERLAP>`** optionally specify the fraction by which cells overlap. The default value is 0.25. If this value is too small, there may be artifacts in the output at the boundary of cells. 
  
#### Additional arguments to `fit.py` and `fit-grid.py`
Additionally, both `fit.py` and `fit-grid.py` accept the following optional arguments which can alter the behavior and performance of
the fitting process:
  * **`--scale <SCALE>`**: Reconstruct the surface in a bounding box whose diameter is --scale times bigger than the diameter of the bounding box of the input points. Defaults is 1.1.
  * **`--regularization <REGULARIZATION>`**: Regularization penalty for kernel ridge regression. Default is 1e-7.
  * **`--nystrom-mode <NYSTROM_MODE>`**: How to generate nystrom samples. Default is 'blue-noise'. Must be one of
    - 'random': choose Nyström samples at random from the input
    - 'blue-noise': downsample the input with blue noise to get Nyström samples
    - 'k-means': use k-means  clustering to generate Nyström samples
  * **`--voxel-downsample-threshold <VOXEL_DOWNSAMPLE_THRESHOLD>`**: If the number of input points is greater than this value, downsample it by averaging points and normals within voxels on a grid. The size of the voxel grid is determined via the --grid-size argument. Default is 150_000.NOTE: This can massively  speed up reconstruction for very large point clouds and generally won't throw away any details.
  * **`--kernel <KERNEL>`**: Which kernel to use. Must be one of 'neural-spline', 'spherical-laplace', or 'linear-angle'. Default is 'neural-spline'.NOTE: The spherical laplace is a good approximation to the neural tangent kernel (see [this paper](https://arxiv.org/pdf/2007.01580.pdf) for details)
  * **`--seed <SEED>`**: Random number generator seed to use.
  * **`--out <OUT>`**:  Path to file to save reconstructed mesh in.
  * **`--save-grid`**: If set, save the function evaluated on a voxel grid to `{out}.grid.npy` where out is the value of the `--out` argument.
  * **`--save-points`**: If set, save the tripled input points, their occupancies, and the Nyström samples to an npz file named `{out}.pts.npz` where out is the value of the `--out` argument.
  * **`--cg-max-iters <CG_MAX_ITERS>`**: Maximum number of conjugate gradient iterations. Default is 20.
  * **`--cg-stop-thresh <CG_STOP_THRESH>`**: Stop threshold for the conjugate gradient algorithm. Default is 1e-5.
  * **`--dtype DTYPE`**: Scalar type of the data. Must be one of 'float32' or 'float64'. Warning: float32 only works for very simple inputs.
  * **`--outer-layer-variance <OUTER_LAYER_VARIANCE>`**: Variance of the outer layer of the neural network from which the neural spline kernel arises from. Default is 1.0.
  * **`--verbose`**: If set, spam your terminal with debug information


## Using Neural Splines in Python
Neural Splines can be used directly from within python by importing the `neural_splines` module in this repository.

To reconstruct a surface using Neural Splines, use the function `neural_splines.fit_model_to_pointcloud`. It returns a `model` object with the same API as Skikit-Learn. *NOTE:* `neural_splines.fit_model_to_pointcloud` can additionally accept other optional arguments. Run `help(neural_splines.fit_model_to_pointcloud)` for details.
```python
from neural_splines import fit_model_to_point_cloud

# x is a point cloud stored in a torch tensor of shape [N, 3]
# n is a tensor of unit normals (one per point) of shape [N, 3]
# num_ny is the number of Nystrom samples to use 
# eps is the finite differencing coefficient (see documentation above)
model = fit_model_to_pointcloud(x, n, num_ny, eps)

# Evaluate the neural spline at a point p
p = torch.tensor([[0.5, 0.5, 0.5]]).to(x)
f_p = model.predict(p)
```

To evaluate a fitted Neural Spline on a grid of points, you can use the function `neural_splines.eval_model_on_grid`. *NOTE:* `neural_splines.eval_model_on_grid` can also accept other optional arguments, run `help(neural_splines.eval_model_on_grid)` for details.
```python
from neural_splines import eval_model_on_grid

# Assume model is a Neural Spline fitted with fit_model_to_point_cloud

# Bounding box of the point cloud x represented as a tuple (origin, size)
bbox = x.min(0)[0], x.max(0)[0] - x.min(0)[0]
grid_res = torch.tensor([128, 128, 128]).to(torch.int32)

recon = eval_model_on_grid(model, bbox, voxel_grid_size)  # a [128, 128, 128] shaped tensor representing the neural spline evaluated on a grid.
```

