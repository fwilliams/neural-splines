import distutils.util
import time

import falkon
import numpy as np
import point_cloud_utils as pcu
import torch

from .falkon_kernels import NeuralSplineKernel, LaplaceKernelSphere, LinearAngleKernel
from .kmeans import kmeans


def make_triples(x, n, eps, homogeneous=False):
    """
    Convert a point cloud equipped with normals into a point cloud with points pertubed along those normals
    and corresponding occupancy values.
    Each point X with normal N, in the input gets converted to 3 points:
        (X, X+eps*N, X-eps*N) with occupancy values (0, 1, -1)
    :param x: The input points of shape [N, 3]
    :param n: The input normals of shape [N, 3]
    :param eps: The amount to perturb points about each normal
    :param homogeneous: If true, return the points in homogeneous coordinates
    :return: A pair, (X, O) consisting of the new point cloud X and point occupancies O
    """
    x_in = x - n * eps
    x_out = x + n * eps
    if isinstance(x, torch.Tensor) and isinstance(n, torch.Tensor):
        x_triples = torch.cat([x, x_in, x_out], dim=0)
        occ_triples = torch.cat([torch.zeros(x.shape[0]),
                                 -torch.ones(x.shape[0]),
                                 torch.ones(x.shape[0])]).to(x) * eps
        if homogeneous:
            x_triples = torch.cat([x_triples, torch.ones(x_triples.shape[0], 1, dtype=x_triples.dtype)], dim=-1)
    else:
        x_triples = np.concatenate([x, x_in, x_out], axis=0)
        occ_triples = np.concatenate([torch.zeros(x.shape[0]),
                                     -torch.ones(x.shape[0]),
                                     torch.ones(x.shape[0])]).astype(x.dtype) * eps
        if homogeneous:
            x_triples = np.concatenate([x_triples, np.ones(x_triples.shape[0], 1, dtype=x_triples.dtype)], axis=-1)
    return x_triples, occ_triples


def points_in_bbox(x, bbox):
    """
    Compute a mask indicating which points in x lie in the bouning box bbox
    :param x: A point cloud represented as a tensor of shape [N, 3]
    :param bbox: A bounding box reprented as 2 3D vectors (origin, size)
    :return: A mask of shape [N] where True values correspond to points in x which lie inside bbox
    """
    mask = torch.logical_and(x > bbox[0], x <= bbox[0] + bbox[1])
    mask = torch.min(mask, axis=-1)[0].to(torch.bool)
    return mask


def normalize_pointcloud_transform(x):
    """
    Compute an affine transformation that normalizes the point cloud x to lie in [-0.5, 0.5]^2
    :param x: A point cloud represented as a tensor of shape [N, 3]
    :return: An affine transformation represented as a tuple (t, s) where t is a translation and s is scale
    """
    min_x, max_x = x.min(0)[0], x.max(0)[0]
    bbox_size = max_x - min_x

    translate = -(min_x + 0.5 * bbox_size)
    scale = 1.0 / torch.max(bbox_size)

    return translate, scale


def affine_transform_pointcloud(x, tx):
    """
    Apply the affine transform tx to the point cloud x
    :param x: A pytorch tensor of shape [N, 3]
    :param tx: An affine transformation represented as a tuple (t, s) where t is a translation and s is scale
    :return: The transformed point cloud
    """
    translate, scale = tx
    return scale * (x + translate)


def affine_transform_bounding_box(bbox, tx):
    """
    Apply the affine transform tx to the bounding box bbox
    :param bbox: A bounding box reprented as 2 3D vectors (origin, size)
    :param tx: An affine transformation represented as a tuple (t, s) where t is a translation and s is scale
    :return: The transformed point bounding box
    """
    translate, scale = tx
    return scale * (bbox[0] + translate), scale * bbox[1]


def voxel_chunks(grid_size, cells_per_axis):
    """
    Iterator over ranges which partition a voxel grid into non-overlapping chunks.
    :param grid_size: Size of the voxel grid to split into chunks
    :param cells_per_axis: Number of cells along each axis
    :return: Each call returns a pair (vmin, vmax) where vmin is the minimum indexes of the voxel chunk and vmax is 
             the maximum index. i.e. if vox is a voxel grid with shape grid_size, then vox[vmin:vmax] are the voxels
             in the current chunk
    """
    if np.isscalar(cells_per_axis):
        cells_per_axis = torch.tensor([cells_per_axis] * len(grid_size)).to(torch.int32)

    current_vox_min = torch.tensor([0.0, 0.0, 0.0]).to(torch.float64)
    current_vox_max = torch.tensor([0.0, 0.0, 0.0]).to(torch.float64)

    cell_size_float = grid_size.to(torch.float64) / cells_per_axis

    for c_i in range(cells_per_axis[0]):
        current_vox_min[0] = current_vox_max[0]
        current_vox_max[0] = cell_size_float[0] + current_vox_max[0]
        current_vox_min[1:] = 0
        current_vox_max[1:] = 0

        for c_j in range(cells_per_axis[1]):
            current_vox_min[1] = current_vox_max[1]
            current_vox_max[1] = cell_size_float[1] + current_vox_max[1]
            current_vox_min[2:] = 0
            current_vox_max[2:] = 0

            for c_k in range(cells_per_axis[2]):
                current_vox_min[2] = current_vox_max[2]
                current_vox_max[2] = cell_size_float[2] + current_vox_max[2]

                vox_min = torch.round(current_vox_min).to(torch.int32)
                vox_max = torch.round(current_vox_max).to(torch.int32)

                yield (c_i, c_j, c_k), vox_min, vox_max


def fit_cell(x, n, cell_bbox, seed, args):
    padded_bbox = scale_bounding_box_diameter(cell_bbox, 1.0 + args.overlap)
    mask = points_in_bbox(x, padded_bbox)
    x, n = x[mask], n[mask]
    x, y = make_triples(x, n, args.eps, homogeneous=False)

    tx = normalize_pointcloud_transform(x)
    x = affine_transform_pointcloud(x, tx)
    x_ny, center_selector, ny_count = generate_nystrom_samples(x, args.num_nystrom_samples, args.nystrom_mode,
                                                               seed, print_message=False)

    x = torch.cat([x, torch.ones(x.shape[0], 1).to(x)], dim=-1)

    model = fit_model(x, y, args.regularization, ny_count, center_selector,
                      maxiters=args.cg_max_iters,
                      kernel_type=args.kernel, stop_thresh=args.cg_stop_thresh,
                      variance=args.outer_layer_variance,
                      verbose=args.verbose, print_message=False)

    return model, tx


def load_point_cloud(filename, min_norm_normal=1e-5, dtype=torch.float64):
    """
    Load a point cloud with normals, filtering out points whose normal has a magnitude below the given threshold.
    :param filename: Path to a PLY file
    :param min_norm_normal: The minimum norm of a normal below which we discard a point
    :param dtype: The output dtype of the tensors returned
    :return: A pair v, n,  where v is a an [N, 3]-shaped tensor of points, n is a [N, 3]-shaped tensor of unit normals
    """
    v, _, n, _ = pcu.read_ply(filename, dtype=np.float64)
    # Some meshes have non unit normals, so build a binary mask of points whose normal has a reasonable magnitude
    # We use this mask to remove bad vertices
    mask = np.linalg.norm(n, axis=-1) > min_norm_normal

    # Keep the good points and normals
    x = v[mask].astype(np.float64)
    n = n[mask].astype(np.float64)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)

    return torch.from_numpy(x).to(dtype), torch.from_numpy(n).to(dtype)


def point_cloud_bounding_box(x, scale=1.0):
    """
    Get the axis-aligned bounding box for a point cloud (possibly scaled by some factor)
    :param x: A point cloud represented as an [N, 3]-shaped tensor
    :param scale: A scale factor by which to scale the bounding box diagonal
    :return: The (possibly scaled) axis-aligned bounding box for a point cloud represented as a pair (origin, size)
    """
    bb_min, bb_size = torch.from_numpy(x.min(0)[0]), torch.from_numpy(x.max(0)[0] - x.min(0)[0])
    bb_diameter = torch.norm(bb_size)
    bb_unit_dir = bb_size / bb_diameter
    scaled_bb_size = bb_size * scale
    scaled_bb_diameter = torch.norm(scaled_bb_size)
    scaled_bb_min = bb_min - 0.5 * (scaled_bb_diameter - bb_diameter) * bb_unit_dir
    return scaled_bb_min, scaled_bb_size


def load_normalized_point_cloud(filename, min_norm_normal=1e-5, dtype=torch.float64):
    v, _, n, _ = pcu.read_ply(filename, dtype=np.float64)
    return normalize_point_cloud(v, n, min_norm_normal, dtype)


def normalize_point_cloud(v, n, min_norm_normal=1e-5, dtype=torch.float64):
    # Some meshes have non unit normals, so build a binary mask of points whose normal has a reasonable magnitude
    # We use this mask to remove bad vertices
    mask = np.linalg.norm(n, axis=-1) > min_norm_normal

    # Keep the good points and normals
    x = v[mask].astype(np.float64)
    n = n[mask].astype(np.float64)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)

    # We need to center and rescale the point cloud in [-0.5, 0.5]^3
    bbox_origin = x.min(0)
    x -= bbox_origin[np.newaxis, :]  # [0, 0, 0] to [sx, sy, sz]
    # print(f"(1) x-range: {x.min(0)}, {x.max(0)}")
    bbox_size = x.max(0) - x.min(0)  # [sz, sy, sz]
    # print(f"(2) x bbox_size: {bbox_size}")
    x -= (bbox_size / 2.0)  # center
    # print(f"(3) x-range: {x.min(0)}, {x.max(0)}")
    x /= bbox_size.max()  # [-0.5, -0.5, -0.5] to up to [0.5, 0.5, 0.5] (aspect ratio is preserved)
    # print(f"(4) x-range: {x.min(0)}, {x.max(0)}")

    n_bbox_origin, n_bbox_size = x.min(0), (x.max(0) - x.min(0))
    x = torch.from_numpy(x).to(dtype)
    n = torch.from_numpy(n).to(dtype)

    # Return points, normals, and transform information to denormalize points
    return x, n, (bbox_origin, bbox_size), (n_bbox_origin, n_bbox_size)


def scale_bounding_box_diameter(bbox, scale):
    bb_min, bb_size = bbox
    bb_diameter = torch.norm(bb_size)
    bb_unit_dir = bb_size / bb_diameter
    scaled_bb_size = bb_size * scale
    scaled_bb_diameter = torch.norm(scaled_bb_size)
    scaled_bb_min = bb_min - 0.5 * (scaled_bb_diameter - bb_diameter) * bb_unit_dir
    return scaled_bb_min, scaled_bb_size


def generate_nystrom_samples(x, num_samples, sampling_method, seed, print_message=True):
    if sampling_method == 'random':
        if print_message:
            print("Using Nyström samples chosen uniformly at random from the input.")
        center_selector = 'uniform'
        x_ny = None
        ny_count = min(num_samples, x.shape[0])
    elif sampling_method == 'blue-noise':
        if print_message:
            print("Generating blue noise Nyström samples.")
        ny_idx = pcu.downsample_point_cloud_poisson_disk(x.numpy(), num_samples, random_seed=seed)
        x_ny = x[ny_idx]
        ny_count = x_ny.shape[0]
        center_selector = falkon.center_selection.FixedSelector(centers=x_ny, y_centers=None)
    elif sampling_method == 'k-means':
        if print_message:
            print("Generating k-means Nyström samples.")
        _, x_ny = kmeans(x.contiguous(), num_samples)
        x_ny = torch.cat([x_ny, torch.ones(x_ny.shape[0], 1).to(x_ny)], dim=-1)
        ny_count = x_ny.shape[0]
        center_selector = falkon.center_selection.FixedSelector(centers=x_ny, y_centers=None)
    else:
        raise ValueError(f"Invalid value {sampling_method} for --nystrom-mode. "
                         f"Must be one of 'random', 'blue-noise' or 'k-means'")

    return x_ny, center_selector, ny_count


def fit_model(x, y, penalty, num_ny, center_selector, kernel_type="neural-spline",
              maxiters=20, stop_thresh=1e-7, variance=1.0, verbose=False, falkon_opts=None, print_message=True):

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
        if print_message:
            print("Using Neural Spline Kernel")
        kernel = NeuralSplineKernel(variance=variance, opt=falkon_opts)
    elif kernel_type == "spherical-laplace":
        if print_message:
            print("Using Spherical Laplace Kernel")
        kernel = LaplaceKernelSphere(alpha=-0.5, gamma=0.5, opt=falkon_opts)
    elif kernel_type == "linear-angle":
        if print_message:
            print("Using Linear Angle Kernel")
        kernel = LinearAngleKernel(opt=falkon_opts)
    else:
        raise ValueError(f"Invalid kernel_type {kernel_type}, expected one of 'neural-spline' or 'spherical-laplace'")

    fit_start_time = time.time()
    model = falkon.Falkon(kernel=kernel, penalty=penalty, M=num_ny, options=falkon_opts, maxiter=maxiters,
                          center_selection=center_selector)

    model.fit(x, y)
    print(f"Fit model in {time.time() - fit_start_time} seconds")
    return model
