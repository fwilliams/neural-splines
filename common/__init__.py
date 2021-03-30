import distutils.util
import time

import falkon
import numpy as np
import point_cloud_utils as pcu
import torch

from .falkon_kernels import NeuralSplineKernel, LaplaceKernelSphere, LinearAngleKernel
from .kmeans import kmeans


def query_yes_no(question, default='no'):
    """
    Prompt the user for a yes or no input. Can accept either y/n, Y/N or yes/no as well as a default response
    :param question: Question to prompt the user.
    :param default: The default response on no input
    :return: A boolean set to True if the user responded yes and False if the user responded no
    """
    if default is None:
        prompt = " [y/n] "
    elif default == 'yes':
        prompt = " [Y/n] "
    elif default == 'no':
        prompt = " [y/N] "
    else:
        raise ValueError(f"Unknown setting '{default}' for default.")

    while True:
        try:
            resp = input(question + prompt).strip().lower()
            if default is not None and resp == '':
                return default == 'yes'
            else:
                return distutils.util.strtobool(resp)
        except ValueError:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


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


def load_normalized_point_cloud(filename, min_norm_normal=1e-5, dtype=torch.float64):
    v, _, n, _ = pcu.read_ply(filename, dtype=np.float64)
    return normalize_point_cloud(v, n, min_norm_normal, dtype)


def load_point_cloud(filename, min_norm_normal=1e-5, dtype=torch.float64):
    v, _, n, _ = pcu.read_ply(filename, dtype=np.float64)
    # Some meshes have non unit normals, so build a binary mask of points whose normal has a reasonable magnitude
    # We use this mask to remove bad vertices
    mask = np.linalg.norm(n, axis=-1) > min_norm_normal

    # Keep the good points and normals
    x = v[mask].astype(np.float64)
    n = n[mask].astype(np.float64)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)

    bbox = torch.from_numpy(x.min(0)), torch.from_numpy(x.max(0) - x.min(0))

    return torch.from_numpy(x), torch.from_numpy(n), bbox


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


def generate_nystrom_samples(x, num_samples, sampling_method, seed):
    if sampling_method == 'random':
        print("Using Nyström samples chosen uniformly at random from the input.")
        center_selector = 'uniform'
        x_ny = None
        ny_count = min(num_samples, x.shape[0])
    elif sampling_method == 'blue-noise':
        print("Generating blue noise Nyström samples.")
        ny_idx = pcu.downsample_point_cloud_poisson_disk(x.numpy(), num_samples, random_seed=seed)
        x_ny = x[ny_idx]
        ny_count = x_ny.shape[0]
        center_selector = falkon.center_selection.FixedSelector(centers=x_ny, y_centers=None)
    elif sampling_method == 'k-means':
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
    elif kernel_type == "linear-angle":
        kernel = LinearAngleKernel(opt=falkon_opts)
    else:
        raise ValueError(f"Invalid kernel_type {kernel_type}, expected one of 'neural-spline' or 'spherical-laplace'")

    fit_start_time = time.time()
    model = falkon.Falkon(kernel=kernel, penalty=penalty, M=num_ny, options=falkon_opts, maxiter=maxiters,
                          center_selection=center_selector)

    model.fit(x, y)
    print(f"Fit model in {time.time() - fit_start_time} seconds")
    return model
