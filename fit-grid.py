import argparse

import numpy as np
import point_cloud_utils as pcu
import torch
import tqdm
from scipy.ndimage import binary_erosion
from skimage.measure import marching_cubes

from neural_splines import load_point_cloud, point_cloud_bounding_box, fit_model_to_pointcloud, eval_model_on_grid, \
    voxel_chunks, points_in_bbox, affine_transform_pointcloud, get_weights


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
    argparser.add_argument("--weight-type", type=str, default='trilinear',
                           help="How to interpolate predictions in overlapping cells. Must be one of 'trilinear' "
                                "or 'none', where 'trilinear' interpolates using a partition of unity defined using a"
                                "bicubic spline and 'none' does not interpolate overlapping cells. "
                                "Default is 'trilinear'.")
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
    out_grid = torch.zeros(*out_grid_size, dtype=torch.float32)
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

        # Amount of voxels by which to pad each cell in each direction
        cell_pad_vox = torch.round(0.5 * args.overlap * out_grid_size.to(torch.float64) / args.cells_per_axis)

        # Minimum and maximum voxel indices of the padded cell
        cell_pvmin = torch.maximum(cell_vmin - cell_pad_vox, torch.zeros(3).to(cell_vmin)).to(torch.int32)
        cell_pvmax = torch.minimum(cell_vmax + cell_pad_vox, torch.tensor(out_grid.shape).to(cell_vmin)).to(torch.int32)

        # Bounding box and point mask for the padded cell
        cell_pad_amount = cell_pad_vox * voxel_size
        padded_cell_bbox = cell_bbox[0] - cell_pad_amount, cell_bbox[1] + 2.0 * cell_pad_amount
        mask_padded_cell = points_in_bbox(x, padded_cell_bbox)

        # Center the cell so it lies in [-0.5, 0.5]^3
        tx = -padded_cell_bbox[0] - 0.5 * padded_cell_bbox[1], 1.0 / torch.max(padded_cell_bbox[1])
        x_cell = x[mask_padded_cell].clone()
        n_cell = n[mask_padded_cell].clone()
        x_cell = affine_transform_pointcloud(x_cell, tx)

        # Cell trilinear blending weights, and index range for which voxels to reconstruct
        weights, idxmin, idxmax = get_weights(cell_vmin, cell_vmax, cell_pvmin, cell_pvmax, args.weight_type)

        # Fit the model and evaluate it on the subset of voxels corresponding to this cell
        cell_model, _ = fit_model_to_pointcloud(x_cell, n_cell,
                                                num_ny=args.num_nystrom_samples, eps=args.eps,
                                                kernel=args.kernel, reg=args.regularization, ny_mode=args.nystrom_mode,
                                                cg_max_iters=args.cg_max_iters, cg_stop_thresh=args.cg_stop_thresh,
                                                outer_layer_variance=args.outer_layer_variance,
                                                verbosity_level=7 if not args.verbose else 0,
                                                normalize=False)
        cell_recon = eval_model_on_grid(cell_model, scaled_bbox, tx, out_grid_size,
                                        cell_vox_min=idxmin, cell_vox_max=idxmax, print_message=False)

        w_cell_recon = weights * cell_recon
        out_grid[idxmin[0]:idxmax[0], idxmin[1]:idxmax[1], idxmin[2]:idxmax[2]] += w_cell_recon
        out_mask[cell_vmin[0]:cell_vmax[0], cell_vmin[1]:cell_vmax[1], cell_vmin[2]:cell_vmax[2]] = True
        tqdm_bar.update(1)

    out_grid[torch.logical_not(out_mask)] = 1.0
    if args.save_grid:
        np.savez(args.out + ".grid", grid=out_grid.detach().cpu().numpy(), mask=out_mask.detach().cpu().numpy())

    # Erode the mask so we don't get weird boundaries
    eroded_mask = binary_erosion(out_mask.numpy().astype(np.bool), np.ones([3, 3, 3]).astype(np.bool))
    v, f, n, c = marching_cubes(out_grid.numpy(), level=0.0, mask=eroded_mask, spacing=voxel_size,
                                gradient_direction='ascent')
    v += scaled_bbox[0].numpy() + 0.5 * voxel_size.numpy()
    pcu.save_mesh_vfn(args.out, v, f, n)


if __name__ == "__main__":
    main()
