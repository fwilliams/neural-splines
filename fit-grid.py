import argparse

import numpy as np
import point_cloud_utils as pcu
import torch
from skimage.measure import marching_cubes

from common import make_triples, load_point_cloud, scale_bounding_box_diameter, \
    generate_nystrom_samples, fit_model


def reconstruct_on_grid(model, full_grid_size, full_bbox, cell_bbox, cell_bbox_normalized, scale, out, out_mask,
                        dtype=torch.float64):
    full_bbmin, full_bbsize = full_bbox
    cell_bbmin, cell_bbsize = cell_bbox

    # print("FULL BBOX", full_bbmin, full_bbsize)
    # print("CELL BBOX", cell_bbmin, cell_bbsize)

    # print("  full_bbsize, full_grid_width", full_bbsize, full_grid_width)
    # full_grid_size = np.round(full_bbsize / scale * full_grid_width).astype(np.int64)
    # print("  full_grid_size", full_grid_size)

    cell_bbmin_rel = (cell_bbmin - full_bbmin) / full_bbsize
    cell_bbmax_rel = (cell_bbmin + cell_bbsize - full_bbmin) / full_bbsize

    # print("RELATIVE BBOX", cell_bbmin_rel, cell_bbmax_rel)

    cell_vox_min = np.round(cell_bbmin_rel * full_grid_size).astype(np.int32)
    cell_vox_max = np.minimum(np.round(cell_bbmax_rel * full_grid_size).astype(np.int32) + 1, full_grid_size)
    print(" ", cell_vox_min, cell_vox_max)
    cell_vox_size = cell_vox_max - cell_vox_min

    plt_range_min, plt_range_max = cell_bbox_normalized[0], cell_bbox_normalized[0] + cell_bbox_normalized[1]
    xgrid = np.stack([_.ravel() for _ in np.mgrid[plt_range_min[0]:plt_range_max[0]:cell_vox_size[0] * 1j,
                                                  plt_range_min[1]:plt_range_max[1]:cell_vox_size[1] * 1j,
                                                  plt_range_min[2]:plt_range_max[2]:cell_vox_size[2] * 1j]],
                     axis=-1)
    xgrid = torch.from_numpy(xgrid).to(dtype)
    xgrid = torch.cat([xgrid, torch.ones(xgrid.shape[0], 1).to(xgrid)], dim=-1).to(dtype)
    # print(full_grid_size, cell_vox_size)
    ygrid = model.predict(xgrid).reshape(tuple(cell_vox_size.astype(np.int)))
    ygrid = ygrid.detach().cpu().numpy()

    print(cell_vox_max - cell_vox_min)
    print(out[cell_vox_min[0]:cell_vox_max[0],
          cell_vox_min[1]:cell_vox_max[1],
          cell_vox_min[2]:cell_vox_max[2]].shape)
    out[cell_vox_min[0]:cell_vox_max[0],
        cell_vox_min[1]:cell_vox_max[1],
        cell_vox_min[2]:cell_vox_max[2]] = ygrid.astype(out.dtype)
    out_mask[cell_vox_min[0]:cell_vox_max[0],
             cell_vox_min[1]:cell_vox_max[1],
             cell_vox_min[2]:cell_vox_max[2]] = True

    # nnz_mask = out[cell_vox_min[0]:cell_vox_max[0],
    #            cell_vox_min[1]:cell_vox_max[1],
    #            cell_vox_min[2]:cell_vox_max[2]] > 0.0
    # out[cell_vox_min[0]:cell_vox_max[0],
    #     cell_vox_min[1]:cell_vox_max[1],
    #     cell_vox_min[2]:cell_vox_max[2]][~nnz_mask] = ygrid[~nnz_mask].astype(out.dtype)
    # out[cell_vox_min[0]:cell_vox_max[0],
    #     cell_vox_min[1]:cell_vox_max[1],
    #     cell_vox_min[2]:cell_vox_max[2]][nnz_mask] += ygrid[nnz_mask].astype(out.dtype)
    # out[cell_vox_min[0]:cell_vox_max[0],
    #     cell_vox_min[1]:cell_vox_max[1],
    #     cell_vox_min[2]:cell_vox_max[2]][nnz_mask] /= 2.0

    return ygrid


def points_in_bbox(x, bbox):
    mask = torch.logical_and(x > bbox[0], x <= bbox[0] + bbox[1])
    mask = torch.min(mask, axis=-1)[0].to(torch.bool)
    return mask


def normalizing_transform(x):
    min_x, max_x = x.min(0)[0], x.max(0)[0]
    bbox_size = max_x - min_x

    translate = -(min_x + 0.5 * bbox_size)
    scale = 1.0 / torch.max(bbox_size)

    return translate, scale


def inverse_affine_transform(tx):
    translate, scale = tx
    return -translate, 1.0 / scale


def affine_transform_point_cloud(x, tx):
    translate, scale = tx
    return scale * (x + translate)


def affine_transform_bounding_box(bbox, tx):
    translate, scale = tx
    return scale * (bbox[0] + translate), scale * bbox[1]


def fit_cell(x, n, cell_bbox, seed, args):
    padded_bbox = scale_bounding_box_diameter(cell_bbox, 1.0 + args.overlap)
    mask = points_in_bbox(x, padded_bbox)
    x, n = x[mask], n[mask]
    x, y = make_triples(x, n, args.eps, homogeneous=False)

    tx = normalizing_transform(x)
    x = affine_transform_point_cloud(x, tx)
    x_ny, center_selector, ny_count = generate_nystrom_samples(x, args.num_nystrom_samples, args.nystrom_mode, seed)

    x = torch.cat([x, torch.ones(x.shape[0], 1).to(x)], dim=-1)

    model = fit_model(x, y, args.regularization, ny_count, center_selector,
                      maxiters=args.cg_max_iters,
                      kernel_type=args.kernel, stop_thresh=args.cg_stop_thresh,
                      variance=args.outer_layer_variance,
                      verbose=args.verbose)
    recon_bbox = affine_transform_bounding_box(cell_bbox, inverse_affine_transform(tx))

    return model, recon_bbox


def eval_cell(model, cell_voxel_size, recon_bbox, dtype):
    recon_bbox = [_.numpy() for _ in recon_bbox]
    cell_voxel_size = cell_voxel_size.numpy()
    xmin, xmax = recon_bbox[0], (recon_bbox[0] + recon_bbox[1])

    xgrid = np.stack([_.ravel() for _ in np.mgrid[xmin[0]:xmax[0]:cell_voxel_size[0] * 1j,
                                                  xmin[1]:xmax[1]:cell_voxel_size[1] * 1j,
                                                  xmin[2]:xmax[2]:cell_voxel_size[2] * 1j]], axis=-1)
    xgrid = torch.from_numpy(xgrid).to(dtype)
    xgrid = torch.cat([xgrid, torch.ones(xgrid.shape[0], 1).to(xgrid)], dim=-1).to(dtype)

    ygrid = model.predict(xgrid).reshape(tuple(cell_voxel_size.astype(np.int))).detach().cpu()

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

    x, n, bbox = load_point_cloud(args.input_point_cloud, dtype=dtype)
    print("x.shape", x.shape, n.shape)
    print(f"x.min: {x.min(0)[0]}, x.max: {x.max(0)[0]}")

    # if x.shape[0] > args.voxel_downsample_threshold:
    #     x, n, _ = pcu.downsample_point_cloud_voxel_grid(1.0 / args.grid_size, x.numpy(), n.numpy())
    #     x, n = torch.from_numpy(x), torch.from_numpy(n)
    print(f"x.min: {x.min(0)[0]}, x.max: {x.max(0)[0]}")

    scaled_bbox = scale_bounding_box_diameter(bbox, 1.0 + args.overlap)
    print("bbox", bbox)
    print("scaled_bbox", scaled_bbox)
    out_grid_size = torch.round(scaled_bbox[1] / scaled_bbox[1].max() * args.grid_size).to(torch.int32)
    voxel_size = scaled_bbox[1] / out_grid_size  # size of one voxel
    out_grid = torch.ones(*out_grid_size, dtype=torch.float32)
    out_mask = torch.zeros(*out_grid_size, dtype=torch.bool)
    print("full grid size is", out_grid_size)

    print(f"Fitting {x.shape[0]} points using {args.cells_per_axis ** 3} cells")

    cell_bboxes = []
    cell_vox_grids = []
    for c_i in range(args.cells_per_axis):
        cell_bboxes.append([])
        cell_vox_grids.append([])
        for c_j in range(args.cells_per_axis):
            cell_bboxes[c_i].append([])
            cell_vox_grids[c_i].append([])
            for c_k in range(args.cells_per_axis):
                cell_idx = torch.tensor((c_i, c_j, c_k)).to(torch.int32)

                # Size of the cell in voxels (pad the last cell with an extra voxel)
                cell_size_float = out_grid_size.to(torch.float64) / args.cells_per_axis
                cell_vox_size = torch.floor(cell_size_float)
                add_one = torch.tensor([1 if c.item() == args.cells_per_axis - 1 else 0 for c in cell_idx])
                cell_vox_size = (cell_vox_size + add_one * torch.ceil(cell_size_float - cell_vox_size)).to(torch.int32)
                cell_vox_origin = (cell_idx * cell_vox_size).to(torch.int32)

                # Bounding box of the cell in world coordinates
                cell_bbox = bbox[0] + cell_vox_origin * voxel_size, cell_vox_size * voxel_size

                cell_bboxes[c_i][c_j].append(cell_bbox)
                cell_vox_grids[c_i][c_j].append((cell_vox_origin, cell_vox_size))

                # If there are no points in this region, then skip it
                mask_cell = points_in_bbox(x, cell_bbox)
                if mask_cell.sum() <= 0:
                    continue

                print(f"Cell {c_i}, {c_j}, {c_k} has size {cell_vox_size} and origin {cell_vox_origin}")
                print(f"    bbox size {cell_bbox[1]}, bbox origin: {cell_bbox[0]}")
                print(f"    x.min: {x.min(0)[0]}, x.max: {x.max(0)[0]}")
                print(f"    num points {mask_cell.sum()}")

                model_ijk, recon_bbox = fit_cell(x, n, cell_bbox, seed, args)

                cell_vox_min, cell_vox_max = cell_vox_origin, cell_vox_origin + cell_vox_size
                out_grid[cell_vox_min[0]:cell_vox_max[0],
                cell_vox_min[1]:cell_vox_max[1],
                cell_vox_min[2]:cell_vox_max[2]] = \
                    eval_cell(model_ijk, cell_vox_size, recon_bbox, dtype).to(out_grid.dtype)
                out_mask[cell_vox_min[0]:cell_vox_max[0],
                cell_vox_min[1]:cell_vox_max[1],
                cell_vox_min[2]:cell_vox_max[2]] = True

    torch.save((cell_bboxes, cell_vox_grids, x), "debug.pth")
    torch.save(out_grid, "out.grid.pth")
    v, f, n, c = marching_cubes(out_grid.numpy(), level=0.0, mask=out_mask.numpy())
    pcu.write_ply(f"recon.ply",
                  v.astype(np.float32), f.astype(np.int32),
                  n.astype(np.float32), c.astype(np.float32))


if __name__ == "__main__":
    main()