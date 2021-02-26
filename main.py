import argparse

import falkon
import numpy as np
import point_cloud_utils as pcu
import time
import torch
from skimage.measure import marching_cubes

from common.falkon_kernels import ArcCosineKernel, LaplaceKernelSphere, NeuralTangentKernel, DirectKernelSolver
from common import make_triples, load_normalized_point_cloud


def fit_model(x, y, penalty, num_ny, kernel_type="spherical-laplace", mode="falkon",
              maxiters=20, stop_thresh=1e-7, verbose=False):
    if isinstance(num_ny, torch.Tensor):
        selector = falkon.center_selection.FixedSelector(centers=num_ny, y_centers=None)
        num_ny = num_ny[0].shape[0]
        print("Using fixed Nystrom samples")
    else:
        selector = 'uniform'

    opts = falkon.FalkonOptions()
    opts.min_cuda_pc_size_64 = 1
    opts.min_cuda_pc_size_32 = 1
    opts.cg_tolerance = stop_thresh
    # opts.cg_full_gradient_every = 10
    opts.debug = verbose
    opts.use_cpu = False
    opts.min_cuda_iter_size_64 = 1
    opts.min_cuda_iter_size_32 = 1

    if kernel_type == "spherical-laplace":
        print("Using Spherical Laplacian Kernel")
        kernel = LaplaceKernelSphere(alpha=-0.5, gamma=0.5, opt=opts)
    elif kernel_type == "arccosine":
        print("Using Arccosine Kernel")
        kernel = ArcCosineKernel(opt=opts)
    elif kernel_type == "ntk":
        kernel = NeuralTangentKernel(variance=1.0, opt=opts)
    else:
        raise ValueError("Invalid kernel_type")

    opts.debug = False

    fit_start_time = time.time()
    if mode == "falkon":
        model = falkon.Falkon(kernel=kernel, penalty=penalty, M=num_ny, options=opts, maxiter=maxiters,
                              center_selection=selector)
    elif mode == "direct":
        model = DirectKernelSolver(kernel=kernel, penalty=penalty)
    else:
        raise ValueError("'mode' must be one of 'direct' or 'falkon'")

    model.fit(x, y)
    print(f"Fit model in {time.time() - fit_start_time} seconds")
    return model


def eval_grid(model, grid_width, input_bbox, normalized_bbox, padding, dtype=torch.float64):
    normalized_bbox_origin, normalized_bbox_size = normalized_bbox

    normalized_bbox_diameter = np.linalg.norm(normalized_bbox_size)
    d = normalized_bbox_size / normalized_bbox_diameter
    half_pad = 0.5 * (padding * normalized_bbox_diameter)
    scaled_bbox_min = normalized_bbox_origin - d * half_pad
    scaled_bbox_max = scaled_bbox_min + normalized_bbox_size * padding
    scaled_bbox_size = scaled_bbox_max - scaled_bbox_min

    # Resolution of voxel grid on which we sample the implicit on
    voxel_grid_dimensions = np.round(grid_width * normalized_bbox_size / normalized_bbox_size.max())
    voxel_size = scaled_bbox_size / voxel_grid_dimensions  # Dimensions of an individual voxel

    print(f"Evaluating function on grid of size "
          f"{voxel_grid_dimensions[0]}x{voxel_grid_dimensions[1]}x{voxel_grid_dimensions[2]}...")
    xgrid = torch.from_numpy(
        np.stack([_.ravel() for _ in
                  np.mgrid[scaled_bbox_min[0]:scaled_bbox_max[0]:voxel_grid_dimensions[0] * 1j,
                           scaled_bbox_min[1]:scaled_bbox_max[1]:voxel_grid_dimensions[1] * 1j,
                           scaled_bbox_min[2]:scaled_bbox_max[2]:voxel_grid_dimensions[2] * 1j]],
                 axis=-1))
    xgrid = torch.cat([xgrid, torch.ones(xgrid.shape[0], 1).to(xgrid)], dim=-1).to(dtype)

    ygrid = model.predict(xgrid).reshape(voxel_grid_dimensions[0],
                                         voxel_grid_dimensions[1],
                                         voxel_grid_dimensions[2])

    v, f, n, vals = marching_cubes(ygrid.detach().cpu().numpy(),
                                   level=0.0, spacing=voxel_size)
    v -= input_bbox[0]
    mesh = v.astype(np.float64), f.astype(np.int32), n.astype(np.float64), vals.astype(np.float64)

    return ygrid, mesh


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_point_cloud", type=str, help="Path to the input point cloud to reconstruct.")
    argparser.add_argument("--out", type=str, default="recon.ply", help="Path to file to save reconstructed mesh in.")

    argparser.add_argument("--kernel", type=str, default="ntk",
                           help="Which kernel to use. Must be one of 'ntk', 'spherical-laplace', or 'arccosine'.")
    argparser.add_argument("--penalty", type=float, default=0.0,
                           help="Regularization penalty for kernel ridge regression.")
    argparser.add_argument("--num-ny", type=int, default=1024,
                           help="Number of Nyström samples for kernel ridge regression.")
    argparser.add_argument("--seed", type=int, default=-1, help="Random number generator seed to use.")

    argparser.add_argument("--scale", type=float, default=1.1,
                           help="Specifies the ratio between the diameter of the cube used for reconstruction and "
                                "the diameter of the samples' bounding cube..")
    argparser.add_argument("--grid-size", "-g", type=int, default=128,
                           help="Size G of the voxel grid to reconstruct on. I.e. we sample the reconstructed "
                                "function on a G x G x G voxel grid.")
    argparser.add_argument("--save-grid", action="store_true",
                           help="If set, save a .npy file with the function evaluated on a voxel grid of "
                                "shape GxGxG where G is the --grid-width argument")

    argparser.add_argument("--mode", type=str, default="falkon",
                           help="Type of solver to use. Must be either 'falkon' for Falkon conjugate gradient or "
                                "'direct' to use a dense LU factorization")
    argparser.add_argument("--eps", type=float, default=0.01,
                           help="Amount to perturb input points around surface to construct occupancy point cloud. "
                                "A reasonable value for this is half the minimum distance between any two points.")
    argparser.add_argument("--blue-noise-nystrom", action="store_true",
                           help="Generate Nystrom samples by doing blue noise down-sampling on the input point cloud")
    argparser.add_argument("--cg-max-iters", type=int, default=20,
                           help="Maximum number of conjugate gradient iterations.")
    argparser.add_argument("--cg-stop-thresh", type=float, default=1e-2, help="Stop threshold for conjugate gradient")
    argparser.add_argument("--verbose", action="store_true", help="Spam your terminal with debug information")
    args = argparser.parse_args()

    dtype = torch.float64

    if args.seed > 0:
        seed = args.seed
    else:
        seed = np.random.randint(2 ** 32 - 1)
    print("Using seed", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # v is in [-0.5, 0.5]^3 (the aspect ratio is preserved, so the longest axis is along [-0.5, 0.5]
    # n is normalized so each normal has unit norm
    x, n, bbox_original, bbox_normalized = load_normalized_point_cloud(args.input_point_cloud)

    # Triple points about the normal for finite differencing
    x, y = make_triples(x, n, args.eps)

    if args.blue_noise_nystrom:
        seed = args.seed if args.seed > 0 else 0
        ny_idx = pcu.prune_point_cloud_poisson_disk(x.numpy(), args.num_ny, random_seed=seed)
        x_ny = x[ny_idx]
        x_ny = torch.cat([x_ny, torch.ones(x_ny.shape[0], 1).to(x_ny)], dim=-1).to(dtype)
        num_ny = x_ny
        ny_count = x_ny.shape[0]
    else:
        num_ny = args.num_ny if args.num_ny > 0 else x.shape[0]
        ny_count = num_ny

    x_homogeneous = torch.cat([x, torch.ones(x.shape[0], 1).to(x)], dim=-1).to(dtype)

    # Permute to randomize nystrom samples
    # TODO: might be unnecesary but I need to check
    perm = torch.randperm(x.shape[0])
    x_homogeneous, y = x_homogeneous[perm], y[perm]

    print(f"Fitting {x_homogeneous.shape[0]} points using {ny_count} Nystrom samples")
    model = fit_model(x_homogeneous, y, args.penalty, num_ny, mode=args.mode,
                      maxiters=args.cg_max_iters, kernel_type=args.kernel,
                      decay=args.decay, stop_thresh=args.cg_stop_thresh,
                      verbose=args.verbose)

    # model, grid_width, input_bbox, normalized_bbox, padding
    grid, mesh = eval_grid(model, args.grid_size, bbox_original, bbox_normalized, args.scale)

    pcu.write_ply(args.out, *mesh)
    if args.save_grid:
        np.save(args.out + ".grid", grid.detach().cpu().numpy())


if __name__ == "__main__":
    main()
