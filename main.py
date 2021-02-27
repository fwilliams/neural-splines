import argparse

import falkon
import numpy as np
import point_cloud_utils as pcu
import time
import torch
from skimage.measure import marching_cubes

from common.falkon_kernels import ArcCosineKernel, LaplaceKernelSphere, DirectKernelSolver
from common import make_triples, load_normalized_point_cloud, scale_bounding_box_diameter


def fit_model(x, y, penalty, num_ny, kernel_type="spherical-laplace", mode="falkon", maxiters=20, decay=-1.0,
              stop_thresh=1e-7, verbose=False):
    if isinstance(num_ny, torch.Tensor):
        selector = falkon.center_selection.FixedSelector(centers=num_ny, y_centers=None)
        num_ny = num_ny[0].shape[0]
        print("Using fixed Nystrom samples")
    else:
        selector = 'uniform'

    opts = falkon.FalkonOptions()
    opts.min_cuda_pc_size_64 = 1
    opts.cg_tolerance = stop_thresh
    # opts.cg_full_gradient_every = 10
    opts.debug = verbose
    opts.use_cpu = False
    opts.min_cuda_iter_size_64 = 1

    if kernel_type == "spherical-laplace":
        if decay > 0.0:
            raise NotImplementedError("TODO: Implement arc cosine kernel with decay")
        else:
            print("Using Spherical Laplacian Kernel")
            kernel = LaplaceKernelSphere(alpha=-1.0, gamma=0.5, opt=opts)
    elif kernel_type == "arccosine":
        if decay > 0.0:
            raise NotImplementedError("TODO: Implement arc cosine kernel with decay")
        else:
            print("Using Arccosine Kernel")
            kernel = ArcCosineKernel(opt=opts)
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


def reconstruct_on_voxel_grid(model, grid_width, scale, bbox_normalized, bbox_input, dtype=torch.float64):
    scaled_bbn_min, scaled_bbn_size = scale_bounding_box_diameter(bbox_normalized, scale)
    scaled_bbi_min, scaled_bbi_size = scale_bounding_box_diameter(bbox_input, scale)

    plt_range_min, plt_range_max = scaled_bbn_min, scaled_bbn_min + scaled_bbn_size
    grid_size = np.round(bbox_normalized[1] * grid_width).astype(np.int64)

    print(f"Evaluating function on grid of size {grid_size[0]}x{grid_size[1]}x{grid_size[2]}...")
    xgrid = np.stack([_.ravel() for _ in np.mgrid[plt_range_min[0]:plt_range_max[0]:grid_size[0] * 1j,
                                                  plt_range_min[1]:plt_range_max[1]:grid_size[1] * 1j,
                                                  plt_range_min[2]:plt_range_max[2]:grid_size[2] * 1j]],
                     axis=-1)
    xgrid = torch.cat([torch.from_numpy(xgrid), torch.ones(xgrid.shape[0], 1).to(xgrid)], dim=-1).to(dtype)

    ygrid = model.predict(xgrid).reshape(grid_size[0], grid_size[1], grid_size[2])

    size_per_voxel = scaled_bbi_size / (grid_size - 1.0)

    v, f, n, vals = marching_cubes(ygrid.detach().cpu().numpy(), level=0.0, spacing=size_per_voxel)
    v += scaled_bbi_min

    return ygrid, (v.astype(np.float64), f.astype(np.int32), n.astype(np.float64), vals.astype(np.float64))


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_point_cloud", type=str, help="Path to the input point cloud to reconstruct.")
    argparser.add_argument("--out", type=str, default="recon.ply", help="Path to file to save reconstructed mesh in.")

    argparser.add_argument("--eps", type=float, default=0.01,
                           help="Amount to perturb input points around surface to construct occupancy point cloud. "
                                "A reasonable value for this is half the minimum distance between any two points.")
    argparser.add_argument("--penalty", type=float, default=0.0,
                           help="Regularization penalty for kernel ridge regression.")
    argparser.add_argument("--num-ny", type=int, default=-1,
                           help="Number of Nyström samples for kernel ridge regression. If negative, don't use "
                                "Nyström sampling")
    argparser.add_argument("--grid-size", "-g", type=int, default=128,
                           help="Size G of the voxel grid to reconstruct on. I.e. we sample the reconstructed "
                                "function on a G x G x G voxel grid.")
    
    argparser.add_argument("--dtype", type=str, default="float64",
                           help="Scalar type of the data. Must be one of 'float32' or 'float64'")

    argparser.add_argument("--kernel", type=str, default="spherical-laplace",
                           help="Which kernel to use. Must be one of 'spherical-laplace' or 'arccosine'.")

    argparser.add_argument("--seed", type=int, default=-1, help="Random number generator seed to use.")

    argparser.add_argument("--scale", type=float, default=1.1,
                           help="If set to a positive value, will normalize the input point cloud so that it is "
                                "centered in a bounding cube of shape [-l, l]^ where l = plot_range - padding. "
                                "Here plot_range is the --plot-range argument and padding is this argument.")

    argparser.add_argument("--save-grid", action="store_true",
                           help="If set, save a .npy file with the function evaluated on a voxel grid of "
                                "shape GxGxG where G is the --grid-width argument")
    argparser.add_argument("--save-points", action="store_true", help="Save input points and Nystrom samples")

    argparser.add_argument("--lloyd-nystrom", action="store_true",
                           help="Generate Nystrom samples by doing lloyd relaxation on the input mesh")

    argparser.add_argument("--cg-max-iters", type=int, default=20,
                           help="Maximum number of conjugate gradient iterations.")
    argparser.add_argument("--cg-stop-thresh", type=float, default=1e-2, help="Stop threshold for conjugate gradient")
    argparser.add_argument("--verbose", action="store_true", help="Spam your terminal with debug information")
    args = argparser.parse_args()

    if args.dtype == "float64":
        dtype = torch.float64
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"invalid --dtype argument. Must be one of 'float32' or 'float64' but got {args.dtype}")

    if args.seed > 0:
        seed = args.seed
    else:
        seed = np.random.randint(2 ** 32 - 1)
    print("Using seed", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    x, n, bbox_input, bbox_normalized = load_normalized_point_cloud(args.input_point_cloud)
    x, y = make_triples(x, n, args.eps)
    x_homogeneous = torch.cat([x, torch.ones(x.shape[0], 1).to(x)], dim=-1).to(dtype)
    y = y.to(x)

    if args.lloyd_nystrom:
        seed = args.seed if args.seed > 0 else 0
        ny_idx = pcu.prune_point_cloud_poisson_disk(x.numpy(), args.num_ny, random_seed=seed)
        x_ny = x_homogeneous[ny_idx]
        num_ny = x_ny
        ny_count = x_ny.shape[0]
    else:
        x_ny = None
        num_ny = args.num_ny if args.num_ny > 0 else x.shape[0]
        ny_count = num_ny

    print(f"Fitting {x_homogeneous.shape[0]} points using {ny_count} Nystrom samples")
    mdl = fit_model(x_homogeneous, y, args.penalty, num_ny, mode="falkon", maxiters=args.cg_max_iters,
                    kernel_type=args.kernel, decay=args.decay, stop_thresh=args.cg_stop_thresh,
                    verbose=args.verbose)
    grid, mesh = reconstruct_on_voxel_grid(mdl, args.grid_size, args.scale, bbox_normalized, bbox_input, dtype=dtype)

    pcu.write_ply(args.out, *mesh)
    if args.save_grid:
        np.savez(args.out + ".grid", grid=grid.detach().cpu().numpy())

    if args.save_points:
        if x_ny is None:
            x_ny = mdl.ny_points_[:, :3] if mdl.ny_points_ is not None else None
        else:
            x_ny = x_ny[:, :3]
        np.savez(args.out + ".pts",
                 x=x.detach().cpu().numpy(),
                 y=y.detach().cpu().numpy(),
                 x_ny=x_ny.detach().cpu().numpy())


if __name__ == "__main__":
    main()
