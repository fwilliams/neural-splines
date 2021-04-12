import argparse
import tqdm

import numpy as np
import point_cloud_utils as pcu
import torch
from skimage.measure import marching_cubes

from neural_splines import load_point_cloud, point_cloud_bounding_box, fit_model_to_pointcloud, eval_model_on_grid, \
    voxel_chunks, points_in_bbox, scale_bounding_box_diameter, affine_transform_pointcloud

from scipy.interpolate import RegularGridInterpolator


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_point_cloud", type=str, help="Path to the input point cloud to reconstruct.")
    argparser.add_argument("eps", type=float,
                           help="Perturbation amount for finite differencing. To approximate the gradient of the "
                                "function, we sample points +/- eps along the normal direction. "
                                "A reasonable value for this is half the minimum distance between any two points.")
    argparser.add_argument("num_nystrom_samples", type=int, default=-1,
                           help="Number of Nyström samples to use for kernel ridge regression. "
                                "If negative, don't use Nyström sampling."
                                "This is the number of basis centers to use to represent the final function. "
                                "If this value is too small, the reconstruction can miss details in the input. "
                                "Values between 10-100 times sqrt(N) (where N = number of input points) are "
                                "generally good depending on the complexity of the input shape.")
    argparser.add_argument("grid_size", type=int,
                           help="When reconstructing the mesh, use this many voxels along the longest side of the "
                                "bounding box. Default is 128.")
    argparser.add_argument("cells_per_axis", type=int,
                           help="Number of cells per axis to split the input along")

    argparser.add_argument("--overlap", type=float, default=0.25,
                           help="By how much should each grid cell overlap as a fraction of the bounding "
                                "box diagonal. Default is 0.25")
    argparser.add_argument("--min-pts-per-cell", type=int, default=0,
                           help="Ignore cells with fewer points than this value. Default is zero.")

    argparser.add_argument("--scale", type=float, default=1.1,
                           help="Reconstruct the surface in a bounding box whose diameter is --scale times bigger than"
                                " the diameter of the bounding box of the input points. Defaults is 1.1.")
    argparser.add_argument("--regularization", type=float, default=1e-7,
                           help="Regularization penalty for kernel ridge regression. Default is 1e-7.")
    argparser.add_argument("--nystrom-mode", type=str, default="blue-noise",
                           help="How to generate nystrom samples. Default is 'k-means'. Must be one of "
                                "(1) 'random': choose Nyström samples at random from the input, "
                                "(2) 'blue-noise': downsample the input with blue noise to get Nyström samples, or "
                                "(3) 'k-means': use k-means clustering to generate Nyström samples. "
                                "Default is 'blue-noise'")
    argparser.add_argument("--voxel-downsample-threshold", type=int, default=150_000,
                           help="If the number of input points is greater than this value, downsample it by "
                                "averaging points and normals within voxels on a grid. The size of the voxel grid is "
                                "determined via the --grid-size argument. Default is 150_000."
                                "NOTE: This can massively speed up reconstruction for very large point clouds and "
                                "generally won't throw away any details.")
    argparser.add_argument("--kernel", type=str, default="neural-spline",
                           help="Which kernel to use. Must be one of 'neural-spline', 'spherical-laplace', or "
                                "'linear-angle'. Default is 'neural-spline'."
                                "NOTE: The spherical laplace is a good approximation to the neural tangent kernel"
                                "(see https://arxiv.org/pdf/2007.01580.pdf for details)")
    argparser.add_argument("--seed", type=int, default=-1, help="Random number generator seed to use.")

    argparser.add_argument("--out", type=str, default="recon.ply", help="Path to file to save reconstructed mesh in.")
    argparser.add_argument("--save-grid", action="store_true",
                           help="If set, save the function evaluated on a voxel grid to {out}.grid.npy "
                                "where out is the value of the --out argument.")
    argparser.add_argument("--save-points", action="store_true",
                           help="If set, save the tripled input points, their occupancies, and the Nyström samples "
                                "to an npz file named {out}.pts.npz where out is the value of the --out argument.")

    argparser.add_argument("--cg-max-iters", type=int, default=20,
                           help="Maximum number of conjugate gradient iterations. Default is 20.")
    argparser.add_argument("--cg-stop-thresh", type=float, default=1e-5,
                           help="Stop threshold for the conjugate gradient algorithm. Default is 1e-5.")

    argparser.add_argument("--dtype", type=str, default="float64",
                           help="Scalar type of the data. Must be one of 'float32' or 'float64'. "
                                "Warning: float32 may not work very well for complicated inputs.")
    argparser.add_argument("--outer-layer-variance", type=float, default=1.0,
                           help="Variance of the outer layer of the neural network from which the neural "
                                "spline kernel arises from. Default is 1.0.")
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
    torch.manual_seed(seed)
    np.random.seed(seed)
    print("Using random seed", seed)

    x, n = load_point_cloud(args.input_point_cloud, dtype=dtype)

    scaled_bbox = point_cloud_bounding_box(x, args.scale)
    out_grid_size = torch.round(scaled_bbox[1] / scaled_bbox[1].max() * args.grid_size).to(torch.int32)
    voxel_size = scaled_bbox[1] / out_grid_size  # size of one voxel

    # Downsample points to grid resolution if there are enough points
    if x.shape[0] > args.voxel_downsample_threshold:
        print("Downsampling input point cloud to voxel resolution.")
        x, n, _ = pcu.downsample_point_cloud_voxel_grid(voxel_size, x.numpy(), n.numpy(),
                                                        min_bound=scaled_bbox[0],
                                                        max_bound=scaled_bbox[0] + scaled_bbox[1])
        x, n = torch.from_numpy(x), torch.from_numpy(n)

    # Voxel grid to store the output
    out_grid = torch.ones(*out_grid_size, dtype=torch.float32)
    out_mask = torch.zeros(*out_grid_size, dtype=torch.bool)

    print(f"Fitting {x.shape[0]} points using {args.cells_per_axis ** 3} cells")

    # Iterate over each grid cell
    tqdm_bar = tqdm.tqdm(total=args.cells_per_axis ** 3)
    for cell_idx, cell_vmin, cell_vmax in voxel_chunks(out_grid_size, args.cells_per_axis):

        tqdm_bar.set_postfix({"Cell": str(cell_idx)})
        # Bounding box of the cell in world coordinates
        cell_vox_size = cell_vmax - cell_vmin
        cell_bbox = scaled_bbox[0] + cell_vmin * voxel_size, cell_vox_size * voxel_size

        # If there are no points in this region, then skip it
        mask_cell = points_in_bbox(x, cell_bbox)
        if mask_cell.sum() <= max(args.min_pts_per_cell, 0):
            tqdm_bar.update(1)
            continue

        # Pad the cell slightly so boundaries agree
        padded_cell_bbox = scale_bounding_box_diameter(cell_bbox, 1.0 + args.overlap)
        mask_padded_cell = points_in_bbox(x, padded_cell_bbox)

        # Center the cell so it lies in [-0.5, 0.5]^3
        tx = -padded_cell_bbox[0] - 0.5 * padded_cell_bbox[1], 1.0 / torch.max(padded_cell_bbox[1])
        x_cell = x[mask_padded_cell].clone()
        n_cell = n[mask_padded_cell].clone()
        x_cell = affine_transform_pointcloud(x_cell, tx)

        # Cell trilinear blending weights
        pbbox_min = torch.maximum(scaled_bbox[0], padded_cell_bbox[0])
        pbbox_max = torch.minimum(scaled_bbox[0] + scaled_bbox[1], padded_cell_bbox[0] + padded_cell_bbox[1])
        pbbox_size = pbbox_max - pbbox_min
        weights, pcell_vmin, pcell_vmax = cell_weights_trilinear((pbbox_min, pbbox_size), cell_bbox, voxel_size)
        print("weights.shape", weights.shape)

        # Fit the model and evaluate it on the subset of voxels corresponding to this cell
        cell_model, _ = fit_model_to_pointcloud(x_cell, n_cell,
                                                num_ny=args.num_nystrom_samples, eps=args.eps,
                                                kernel=args.kernel, reg=args.regularization, ny_mode=args.nystrom_mode,
                                                cg_max_iters=args.cg_max_iters, cg_stop_thresh=args.cg_stop_thresh,
                                                outer_layer_variance=args.outer_layer_variance,
                                                verbosity_level=7 if not args.verbose else 0,
                                                normalize=False)
        cell_recon = eval_model_on_grid(cell_model, scaled_bbox, tx, out_grid_size,
                                        cell_vox_min=pcell_vmin, cell_vox_max=pcell_vmax, print_message=False)
        print("cell_recon.shape", cell_recon.shape)

        w_cell_recon = weights * cell_recon
        print("w_cell_recon.shape", w_cell_recon.shape)
        print(out_grid[pcell_vmin[0]:pcell_vmax[0], pcell_vmin[1]:pcell_vmax[1], pcell_vmin[2]:pcell_vmax[2]].shape)
        print(pcell_vmin, pcell_vmax, pcell_vmax-pcell_vmin)
        out_grid[pcell_vmin[0]:pcell_vmax[0], pcell_vmin[1]:pcell_vmax[1], pcell_vmin[2]:pcell_vmax[2]] += w_cell_recon
        out_mask[cell_vmin[0]:cell_vmax[0], cell_vmin[1]:cell_vmax[1], cell_vmin[2]:cell_vmax[2]] = True
        tqdm_bar.update(1)

    if args.save_grid:
        np.savez(args.out + ".grid", grid=out_grid.detach().cpu().numpy(), mask=out_mask.detach().cpu().numpy())

    v, f, n, c = marching_cubes(out_grid.numpy(), level=0.0, mask=out_mask.numpy(), spacing=voxel_size)
    v += scaled_bbox[0].numpy() + 0.5 * voxel_size.numpy()
    pcu.save_mesh_vfn(args.out, v, f, n)


def cell_weights_trilinear(padded_cell_bbox, cell_bbox, voxel_size, total_bbox):
    pbmin, pbmax = padded_cell_bbox[0], padded_cell_bbox[0] + padded_cell_bbox[1]
    bmin, bmax = cell_bbox[0], cell_bbox[0] + cell_bbox[1]
    x, y, z = [np.unique(np.array([pbmin[i], bmin[i], bmax[i], pbmax[i]])) for i in range(3)]
    vals = np.zeros([x.shape[0], y.shape[0], z.shape[0]])
    xyz = (x, y, z)

    one_idxs = []
    for dim in range(3):
        if xyz[dim].shape[0] == 2:
            one_idxs.append([0, 1])
        elif xyz[dim].shape[0] == 3:
            if padded_cell_bbox[0][dim] == cell_bbox[0][dim]:
                one_idxs.append([0, 1])
            else:
                one_idxs.append([1, 2])
        else:
            one_idxs.append([1, 2])

    for i in one_idxs[0]:
        for j in one_idxs[1]:
            for k in one_idxs[2]:
                vals[i, j, k] = 1.0
    f_w = RegularGridInterpolator((x, y, z), vals)

    padded_cell_vmin = torch.round(pbmin / voxel_size).to(torch.int32)
    padded_cell_vmax = torch.round(pbmax / voxel_size).to(torch.int32)

    psize = (padded_cell_vmax - padded_cell_vmin).numpy() * 1j
    pmin = ((padded_cell_vmin + 0.5) * voxel_size).numpy()
    pmax = ((padded_cell_vmax - 0.5) * voxel_size).numpy()
    pts = np.stack([np.ravel(a) for a in
                    np.mgrid[pmin[0]:pmax[0]:psize[0], pmin[1]:pmax[1]:psize[1], pmin[2]:pmax[2]:psize[2]]], axis=-1)

    padded_cell_min = torch.round((pbmin - total_bbox[0]) / voxel_size).to(torch.int32)
    padded_cell_size = (padded_cell_vmax - padded_cell_vmin).numpy()
    padded_cell_max = padded_cell_min + padded_cell_size
    return torch.from_numpy(f_w(pts).reshape(padded_cell_size)), padded_cell_min, padded_cell_max


if __name__ == "__main__":
    main()