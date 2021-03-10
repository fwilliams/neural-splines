import argparse

import falkon
import numpy as np
import point_cloud_utils as pcu
import time
import torch
from skimage.measure import marching_cubes
import open3d as o3d

from common.falkon_kernels import LaplaceKernelSphere, NeuralSplineKernel
from common.kmeans import kmeans
from common import make_triples, load_normalized_point_cloud, scale_bounding_box_diameter


def fit_model(x, y, penalty, num_ny, center_selector, kernel_type="spherical-laplace",
              maxiters=20, stop_thresh=1e-7, variance=1.0, verbose=False, falkon_opts=None):

    if falkon_opts is None:
        falkon_opts = falkon.FalkonOptions()

        # Always use cuda for everything
        falkon_opts.min_cuda_pc_size_64 = 1
        falkon_opts.min_cuda_pc_size_32 = 1
        falkon_opts.min_cuda_iter_size_64 = 1
        falkon_opts.min_cuda_iter_size_32 = 1
        falkon_opts.use_cpu = False

        falkon_opts.cg_tolerance = stop_thresh
        falkon_opts.debug = verbose

    if kernel_type == "neural-spline":
        print("Using Neural Spline Kernel")
        kernel = NeuralSplineKernel(variance=variance, opt=falkon_opts)
    elif kernel_type == "spherical-laplace":
        print("Using Spherical Laplace Kernel")
        kernel = LaplaceKernelSphere(alpha=-0.5, gamma=0.5, opt=falkon_opts)
    else:
        raise ValueError(f"Invalid kernel_type {kernel_type}, expected one of 'neural-spline' or 'spherical-laplace'")

    fit_start_time = time.time()
    model = falkon.Falkon(kernel=kernel, penalty=penalty, M=num_ny, options=falkon_opts, maxiter=maxiters,
                          center_selection=center_selector)

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
    xgrid = torch.from_numpy(xgrid).to(dtype)
    xgrid = torch.cat([xgrid, torch.ones(xgrid.shape[0], 1).to(xgrid)], dim=-1).to(dtype)

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
    argparser.add_argument("--nystrom-mode", type=str, default="random",
                           help="How to generate nystrom samples. Must be either (1) "
                                "'random': choose Nyström samples at random from the input, (2)"
                                "'blue-noise': downsample the input with blue noise to get Nyström samples, or (3)"
                                "'k-means': use k-means clustering to generate Nyström samples "
                                "(Warning: K-means is very slow on large inputs)")
    argparser.add_argument("--grid-size", "-g", type=int, default=128,
                           help="Size G of the voxel grid to reconstruct on. I.e. we sample the reconstructed "
                                "function on a G x G x G voxel grid.")
    argparser.add_argument("--voxel-downsample", action="store_true",
                           help="Downsample input point cloud by averaging points and normals "
                                "within a voxel grid, uses the "
                                "--grid-size argument to determine the size of the grid. "
                                "This can massively speed up reconstruction for very large point clouds")
    argparser.add_argument("--dtype", type=str, default="float64",
                           help="Scalar type of the data. Must be one of 'float32' or 'float64'. "
                                "Warning: float32 may not work very well for complicated inputs.")

    argparser.add_argument("--kernel", type=str, default="neural-spline",
                           help="Which kernel to use. Must be one of 'neural-spline' or 'spherical-laplace'."
                                "The spherical laplace is a good approximation to the Neural Tangent Kernel"
                                "(see https://arxiv.org/pdf/2007.01580.pdf for details)")

    argparser.add_argument("--seed", type=int, default=-1, help="Random number generator seed to use.")

    argparser.add_argument("--scale", type=float, default=1.1,
                           help="If set to a positive value, will normalize the input point cloud so that it is "
                                "centered in a bounding cube of shape [-l, l]^ where l = plot_range - padding. "
                                "Here plot_range is the --plot-range argument and padding is this argument.")

    argparser.add_argument("--save-grid", action="store_true",
                           help="If set, save a .npy file with the function evaluated on a voxel grid of "
                                "shape GxGxG where G is the --grid-width argument")
    argparser.add_argument("--save-points", action="store_true", help="Save input points and Nystrom samples")

    argparser.add_argument("--cg-max-iters", type=int, default=20,
                           help="Maximum number of conjugate gradient iterations.")
    argparser.add_argument("--cg-stop-thresh", type=float, default=1e-2, help="Stop threshold for conjugate gradient")

    argparser.add_argument("--outer-layer-variance", type=float, default=1.0,
                           help="Variance of the outer layer of the neural network from which the neural "
                                "spline kernel arises from")
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

    x, n, bbox_input, bbox_normalized = load_normalized_point_cloud(args.input_point_cloud, dtype=dtype)
    if args.voxel_downsample:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x.numpy())
        pcd.normals = o3d.utility.Vector3dVector(n.numpy())
        pc_down = pcd.voxel_down_sample(voxel_size=1.0/args.grid_size)
        x = torch.from_numpy(np.asarray(pc_down.points)).to(dtype)
        n = torch.from_numpy(np.asarray(pc_down.normals)).to(dtype)
    x, y = make_triples(x, n, args.eps)
    x_homogeneous = torch.cat([x, torch.ones(x.shape[0], 1).to(x)], dim=-1)

    if args.nystrom_mode == 'random':
        print("Using Nyström samples chosen uniformly at random from the input...")
        center_selector = 'uniform'
        x_ny = None
        ny_count = args.num_ny
    elif args.nystrom_mode == 'blue-noise':
        print("Generating blue noise Nyström samples...")
        ny_idx = pcu.prune_point_cloud_poisson_disk(x.numpy(), args.num_ny, random_seed=seed)
        x_ny = x_homogeneous[ny_idx]
        ny_count = x_ny.shape[0]
        center_selector = falkon.center_selection.FixedSelector(centers=x_ny, y_centers=None)
    elif args.nystrom_mode == 'k-means':
        print("Generating k-means Nyström samples...")
        _, x_ny = kmeans(x, args.num_ny)
        x_ny = torch.cat([x_ny, torch.ones(x_ny.shape[0], 1).to(x_ny)], dim=-1)
        ny_count = x_ny.shape[0]
        center_selector = falkon.center_selection.FixedSelector(centers=x_ny, y_centers=None)
        torch.save((x, x_ny), "nystrom.pth")
    else:
        raise ValueError(f"Invalid value {args.nystrom_mode} for --nystrom-mode. "
                         f"Must be one of 'random', 'blue-noise' or 'k-means'")

    print(f"Fitting {x_homogeneous.shape[0]} points using {ny_count} Nyström samples")
    mdl = fit_model(x_homogeneous, y, args.penalty, ny_count, center_selector, maxiters=args.cg_max_iters,
                    kernel_type=args.kernel, stop_thresh=args.cg_stop_thresh,
                    variance=args.outer_layer_variance,
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
