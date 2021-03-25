import argparse

import numpy as np
import point_cloud_utils as pcu
import torch
from skimage.measure import marching_cubes

from common import make_triples, load_normalized_point_cloud, scale_bounding_box_diameter, \
    normalize_point_cloud, generate_nystrom_samples, fit_model


# def reconstruct_on_voxel_grid(model, grid_width, scale, bbox_normalized, bbox_input, dtype=torch.float64):
#     scaled_bbn_min, scaled_bbn_size = scale_bounding_box_diameter(bbox_normalized, scale)
#     scaled_bbi_min, scaled_bbi_size = scale_bounding_box_diameter(bbox_input, scale)
#
#     plt_range_min, plt_range_max = scaled_bbn_min, scaled_bbn_min + scaled_bbn_size
#     grid_size = np.round(bbox_normalized[1] * grid_width).astype(np.int64)
#
#     print(f"Evaluating reconstructed function on grid of size {grid_size[0]}x{grid_size[1]}x{grid_size[2]}...")
#     xgrid = np.stack([_.ravel() for _ in np.mgrid[plt_range_min[0]:plt_range_max[0]:grid_size[0] * 1j,
#                                                   plt_range_min[1]:plt_range_max[1]:grid_size[1] * 1j,
#                                                   plt_range_min[2]:plt_range_max[2]:grid_size[2] * 1j]],
#                      axis=-1)
#     xgrid = torch.from_numpy(xgrid).to(dtype)
#     xgrid = torch.cat([xgrid, torch.ones(xgrid.shape[0], 1).to(xgrid)], dim=-1).to(dtype)
#
#     ygrid = model.predict(xgrid).reshape(grid_size[0], grid_size[1], grid_size[2])
#
#     size_per_voxel = scaled_bbi_size / (grid_size - 1.0)
#
#     v, f, n, vals = marching_cubes(ygrid.detach().cpu().numpy(), level=0.0, spacing=size_per_voxel)
#     v += scaled_bbi_min
#
#     return ygrid, (v.astype(np.float64), f.astype(np.int32), n.astype(np.float64), vals.astype(np.float64))


def reconstruct_on_grid(model, full_grid_width, full_bbox, cell_bbox, cell_bbox_normalized, dtype=torch.float64):
    full_bbmin, full_bbsize = full_bbox
    cell_bbmin, cell_bbsize = cell_bbox

    full_grid_size = np.round(full_bbsize * full_grid_width).astype(np.int64)

    cell_bbmin_rel = cell_bbmin - full_bbmin
    cell_bbmax_rel = cell_bbmin_rel + cell_bbsize
    cell_vox_min = np.ceil(cell_bbmax_rel * full_grid_size)
    cell_vox_max = np.floor(cell_bbmax_rel * full_grid_size)
    cell_vox_size = cell_vox_max - cell_vox_min

    plt_range_min, plt_range_max = cell_bbox_normalized[0], cell_bbox_normalized[0] + cell_bbox_normalized[1]
    xgrid = np.stack([_.ravel() for _ in np.mgrid[plt_range_min[0]:plt_range_max[0]:cell_vox_size[0] * 1j,
                                                  plt_range_min[1]:plt_range_max[1]:cell_vox_size[1] * 1j,
                                                  plt_range_min[2]:plt_range_max[2]:cell_vox_size[2] * 1j]],
                     axis=-1)
    xgrid = torch.from_numpy(xgrid).to(dtype)
    xgrid = torch.cat([xgrid, torch.ones(xgrid.shape[0], 1).to(xgrid)], dim=-1).to(dtype)
    ygrid = model.predict(xgrid).reshape(cell_vox_size[0], cell_vox_size[1], cell_vox_size[2])
    ygrid = ygrid.detach().cpu().numpy()

    return ygrid


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

    argparser.add_argument("--overlap", type=float, default=0.1,
                           help="By how much should each grid cell overlap as a fraction of the bounding "
                                "box diagonal. Default is 0.1")

    argparser.add_argument("--regularization", type=float, default=1e-7,
                           help="Regularization penalty for kernel ridge regression. Default is 1e-7.")
    argparser.add_argument("--nystrom-mode", type=str, default="k-means",
                           help="How to generate nystrom samples. Default is 'k-means'. Must be one of "
                                "(1) 'random': choose Nyström samples at random from the input, "
                                "(2) 'blue-noise': downsample the input with blue noise to get Nyström samples, or "
                                "(3) 'k-means': use k-means clustering to generate Nyström samples. "
                                "Default is 'k-means'")
    argparser.add_argument("--voxel-downsample-threshold", type=int, default=150_000,
                           help="If the number of input points is greater than this value, downsample it by "
                                "averaging points and normals within voxels on a grid. The size of the voxel grid is "
                                "determined via the --grid-size argument. Default is 150_000."
                                "NOTE: This can massively speed up reconstruction for very large point clouds and "
                                "generally won't throw away any details.")
    argparser.add_argument("--kernel", type=str, default="neural-spline",
                           help="Which kernel to use. Must be one of 'neural-spline' or 'spherical-laplace'. "
                                "Default is 'neural-spline'."
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
    print("Using seed", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    x, n, bbox_input, bbox_normalized = load_normalized_point_cloud(args.input_point_cloud, dtype=dtype)
    if x.shape[0] > args.voxel_downsample_threshold:
        x, n, _ = pcu.downsample_point_cloud_voxel_grid(1.0 / args.grid_size, x.numpy(), n.numpy())
        x, n = torch.from_numpy(x), torch.from_numpy(n)

    fitted_models = []

    # We're going to include the overlap padding in the final reconstruction.
    # TODO: Do a better version of this where we just include the overlap in the boundary cells
    scaled_bbn_min, scaled_bbn_size = scale_bounding_box_diameter(bbox_normalized, 1.0 + args.overlap)
    cell_bb_size = scaled_bbn_size / args.cells_per_axis

    count = 0
    for cell_i in range(args.cells_per_axis):
        for cell_j in range(args.cells_per_axis):
            for cell_k in range(args.cells_per_axis):
                cell_bb_origin = np.array([cell_i, cell_j, cell_k]) * cell_bb_size
                cell_pad_bb_origin, cell_pad_bb_size = scale_bounding_box_diameter((cell_bb_origin, cell_bb_size),
                                                                                   1.0 + args.overlap)
                mask_ijk = np.logical_and(x > torch.from_numpy(cell_pad_bb_origin),
                                          x <= torch.from_numpy(cell_pad_bb_origin + cell_pad_bb_size))
                print(mask_ijk)
                mask_ijk = torch.min(mask_ijk, axis=-1)[0]
                print(mask_ijk)
                print((mask_ijk > 0).sum())
                x_ijk, n_ijk = x[mask_ijk], n[mask_ijk]
                x_ijk, n_ijk, bbox_ijk, bbox_normalized_ijk = normalize_point_cloud(x_ijk, n_ijk, dtype=dtype)

                x_ijk, y_ijk = make_triples(x_ijk, n_ijk, args.eps)
                x_homogeneous_ijk = torch.cat([x_ijk, torch.ones(x_ijk.shape[0], 1).to(x_ijk)], dim=-1)
                x_ny_ijk, center_selector_ijk, ny_count_ijk = generate_nystrom_samples(x_homogeneous_ijk,
                                                                                       args.num_nystrom_samples,
                                                                                       args.nystrom_mode,
                                                                                       seed)
                print(f"Fitting model ({cell_i}, {cell_j}, {cell_k}) with {x_homogeneous_ijk.shape[0]} points "
                      f"using {ny_count_ijk} Nyström samples.")
                mdl_ijk = fit_model(x_homogeneous_ijk, y_ijk, args.regularization, ny_count_ijk, center_selector_ijk,
                                    maxiters=args.cg_max_iters,
                                    kernel_type=args.kernel, stop_thresh=args.cg_stop_thresh,
                                    variance=args.outer_layer_variance,
                                    verbose=args.verbose)
                fitted_models.append(mdl_ijk.to('cpu'))
                count += 1
                # model, full_grid_width, full_bbox, cell_bbox, cell_bbox_normalized
                # TODO: Saving is for debug only, still need to do full reconstruction
                ygrid = reconstruct_on_grid(mdl_ijk, args.grid_width,
                                            full_bbox=(scaled_bbn_min, scaled_bbn_size),
                                            cell_bbox=(cell_bb_origin, cell_bb_size),
                                            cell_bbox_normalized=bbox_normalized_ijk)
                v_ijk, f_ijk, n_ijk, c_ijk = marching_cubes(ygrid, level=0.0)

                pcu.write_ply(f"recon_{cell_i}_{cell_j}_{cell_k}.ply",
                              v_ijk.astype(np.float32), f_ijk.astype(np.int32),
                              n_ijk.astype(np.float32), c_ijk.astype(np.float32))
                torch.save((mdl_ijk, x_ijk, y_ijk, x_ny_ijk), f"checkpoint_{cell_i}_{cell_j}_{cell_k}.pth")


if __name__ == "__main__":
    main()