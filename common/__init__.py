import time

import falkon
import point_cloud_utils as pcu

from .falkon_kernels import NeuralSplineKernel, LaplaceKernelSphere, LinearAngleKernel
from .geometry import *
from .kmeans import kmeans


def _generate_nystrom_samples(x, num_samples, sampling_method, seed, print_message=True):
    if x.shape[1] != 3:
        raise ValueError(f"Invalid shape for x, must be [N, 3] but got {x.shape}")
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
        x_ny = torch.cat([x_ny, torch.ones(x_ny.shape[0], 1).to(x_ny)], dim=-1)
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


def _run_falkon_fit(x, y, penalty, num_ny, center_selector, kernel_type="neural-spline",
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
    if print_message:
        print(f"Fit model in {time.time() - fit_start_time} seconds")
    return model


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


def fit_model_to_pointcloud(x, n, args, seed=0, print_message=True):
    """
    Fit a kernel to the point cloud with points x and normals n.
    :param x: A tensor of 3D points with shape [N, 3]
    :param n: A tensor of unit normals with shape [N, 3]
    :param seed: If greater than 0, use this random seed to generate Nystrom samples.
    :param args: Arguments for the fitting process. See command line arguments for fit.py and fit-grid.py
    :param print_message: If true, print status messages to stdout.
    :return: A pair (model, tx) where model is a fitted neural spline model class (with the same API as scikit-learn)
             and tx is an affine transformation which converts world space samples to model coordinates.
             You *must* apply this transformation to points before evaluating the model.
             This transformation is represented as a tuple (t, s) where t is a translation and s is scale.
    """

    if seed <= 0:
        seed = np.random.randint(2 ** 32 - 1)
        torch.manual_seed(seed)
        np.random.seed(seed)

    x, y = triple_points_along_normals(x, n, args.eps, homogeneous=False)

    tx = normalize_pointcloud_transform(x)
    x = affine_transform_pointcloud(x, tx)
    x_ny, center_selector, ny_count = _generate_nystrom_samples(x, args.num_nystrom_samples, args.nystrom_mode,
                                                                seed, print_message=print_message)

    x = torch.cat([x, torch.ones(x.shape[0], 1).to(x)], dim=-1)

    model = _run_falkon_fit(x, y, args.regularization, ny_count, center_selector,
                            maxiters=args.cg_max_iters,
                            kernel_type=args.kernel, stop_thresh=args.cg_stop_thresh,
                            variance=args.outer_layer_variance,
                            verbose=args.verbose, print_message=print_message)

    return model, tx


def eval_model_on_grid(model, bbox, tx, voxel_grid_size, cell_vox_min=None, cell_vox_max=None, print_message=True):
    """
    Evaluate the trained model (output of fit_model_to_pointcloud) on a voxel grid.
    :param model: The trained model returned from fit_model_to_pointcloud
    :param bbox: The bounding box defining the region of space on which to evaluate the model
                 (represented as the pair (origin, size))
    :param tx: An affine transformation which transforms points in world coordinates to model
               coordinates before evaluating the model (the second return value of fit_model_to_grid).
               The transformation is represented as a tuple (t, s) where t is a translation and s is scale.
    :param voxel_grid_size: The size of the voxel grid on which to reconstruct
    :param cell_vox_min: If not None, reconstruct on the subset of the voxel grid starting at these indices.
    :param cell_vox_max: If not None, reconstruct on the subset of the voxel grid ending at these indices.
    :param print_message: If true, print status messages to stdout.
    :return: A tensor representing the model evaluated on a grid.
    """
    bbox_origin, bbox_size = bbox
    voxel_size = bbox_size / voxel_grid_size  # size of a single voxel cell

    if cell_vox_min is None:
        cell_vox_min = torch.tensor([0, 0, 0], dtype=torch.int32)

    if cell_vox_max is None:
        cell_vox_max = voxel_grid_size

    if print_message:
        print(f"Evaluating model on grid of size {[_.item() for _ in (cell_vox_max - cell_vox_min)]}.")
    eval_start_time = time.time()

    xmin = bbox_origin + (cell_vox_min + 0.5) * voxel_size
    xmax = bbox_origin + (cell_vox_max - 0.5) * voxel_size

    xmin = affine_transform_pointcloud(xmin.unsqueeze(0), tx).squeeze()
    xmax = affine_transform_pointcloud(xmax.unsqueeze(0), tx).squeeze()

    xmin, xmax = xmin.numpy(), xmax.numpy()
    cell_vox_size = (cell_vox_max - cell_vox_min).numpy()

    xgrid = np.stack([_.ravel() for _ in np.mgrid[xmin[0]:xmax[0]:cell_vox_size[0] * 1j,
                                                  xmin[1]:xmax[1]:cell_vox_size[1] * 1j,
                                                  xmin[2]:xmax[2]:cell_vox_size[2] * 1j]], axis=-1)
    xgrid = torch.from_numpy(xgrid).to(model.alpha_.dtype)
    xgrid = torch.cat([xgrid, torch.ones(xgrid.shape[0], 1).to(xgrid)], dim=-1).to(model.alpha_.dtype)

    ygrid = model.predict(xgrid).reshape(tuple(cell_vox_size.astype(np.int))).detach().cpu()

    if print_message:
        print(f"Evaluated model in {time.time() - eval_start_time}s.")

    return ygrid






