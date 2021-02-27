import distutils.util

import numpy as np
import point_cloud_utils as pcu
import torch


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


def make_triples(x, n, eps):
    """
    Convert a point cloud equipped with normals into a point cloud with points pertubed along those normals
    and corresponding occupancy values.
    Each point X with normal N, in the input gets converted to 3 points:
        (X, X+eps*N, X-eps*N) with occupancy values (0, 1, -1)
    :param x: The input points of shape [N, 3]
    :param n: The input normals of shape [N, 3]
    :param eps: The amount to perturb points about each normal
    :return: A pair, (X, O) consisting of the new point cloud X and point occupancies O
    """
    x_in = x - n * eps
    x_out = x + n * eps
    if isinstance(x, torch.Tensor) and isinstance(n, torch.Tensor):
        x_triples = torch.cat([x, x_in, x_out], dim=0)
        occ_triples = torch.cat([torch.zeros(x.shape[0]),
                                 -torch.ones(x.shape[0]),
                                 torch.ones(x.shape[0])]) * eps
    else:
        x_triples = np.concatenate([x, x_in, x_out], axis=0)
        occ_triples = np.concatenate([torch.zeros(x.shape[0]),
                                     -torch.ones(x.shape[0]),
                                     torch.ones(x.shape[0])]) * eps
    return x_triples, occ_triples


def load_normalized_point_cloud(filename, min_norm_normal=1e-5):
    v, f, n, _ = pcu.read_ply(filename, dtype=np.float64)

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
    x /= bbox_size.max()  # [0, 0, 0] to up to [1, 1, 1] (aspect ratio is preserved)
    # print(f"(4) x-range: {x.min(0)}, {x.max(0)}")

    n_bbox_origin, n_bbox_size = x.min(0), (x.max(0) - x.min(0))
    x = torch.from_numpy(v[mask]).to(torch.float64)
    n = torch.from_numpy(n[mask]).to(torch.float64)

    # Return points, normals, and transform information to denormalize points
    return x, n, (bbox_origin, bbox_size), (n_bbox_origin, n_bbox_size)
