import argparse

import numpy as np
import point_cloud_utils as pcu
import torch

from neural_splines.geometry import point_cloud_bounding_box


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_points", type=str)
    argparser.add_argument("mesh", type=str)
    argparser.add_argument("grid_size", type=int,
                           help="When trimming the mesh, use this many voxels along the longest side of the "
                                "bounding box. This is used to determine the size of a voxel and "
                                "hence the units of distance to use. You should set this to the save value you used in "
                                "fit.py or fit-grid.py")
    argparser.add_argument("trim_distance", type=float,
                           help="Trim vertices of the reconstructed mesh whose nearest "
                                "point in the input is greater than this value. The units of this argument are voxels "
                                "(where the cells_per_axis determines the size of a voxel) Default is -1.0.")
    argparser.add_argument("--out", type=str, default="trimmed.ply", help="Path to file to save trim mesh to.")
    argparser.add_argument("--use-abs-units", action="store_true",
                           help="If set, then use absolute units instead of voxel units for the trim distance.")
    args = argparser.parse_args()

    print(f"Loading input point cloud {args.input_points}")
    p = pcu.load_mesh_v(args.input_points)
    scaled_bbox = point_cloud_bounding_box(torch.from_numpy(p), args.scale)
    out_grid_size = np.round(scaled_bbox[1].numpy() / scaled_bbox[1].max().item() * args.grid_size).astype(np.int32)
    voxel_size = scaled_bbox[1] / out_grid_size  # size of one voxel

    print(f"Loading reconstructed mesh {args.mesh}")
    v, f, n = pcu.load_mesh_vfn(args.mesh)

    print("Trimming mesh...")
    # Trim distance in world coordinates
    if args.use_abs_units:
        trim_dist_world = args.trim_distance
    else:
        trim_dist_world = args.trim_distance * torch.norm(voxel_size).item()
    nn_dist, _ = pcu.k_nearest_neighbors(v, p, k=2)
    nn_dist = nn_dist[:, 1]
    f_mask = np.stack([nn_dist[f[:, i]] < trim_dist_world for i in range(f.shape[1])], axis=-1)
    f_mask = np.all(f_mask, axis=-1)
    f = f[f_mask]

    print("Saving trimmed mesh...")
    pcu.save_mesh_vfn(args.out, v, f, n)
    print("Done!")


if __name__ == "__main__":
    main()
