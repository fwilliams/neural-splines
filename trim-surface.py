import open3d as o3d
import argparse
import point_cloud_utils as pcu
import numpy as np
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
    args = argparser.parse_args()

    print(f"Loading input point cloud {args.input_points}")
    p = pcu.load_mesh_v(args.input_points)
    scaled_bbox = point_cloud_bounding_box(x, args.scale)
    out_grid_size = np.round(scaled_bbox[1].numpy() / scaled_bbox[1].max().item() * args.grid_size).to(np.int32)
    voxel_size = scaled_bbox[1] / out_grid_size  # size of one voxel

    print(f"Loading reconstructed mesh {args.mesh}")
    v, f, n = pcu.load_mesh_vfn(args.mesh)

    print("Trimming mesh...")
    nn_dist, _ = pcu.k_nearest_neighbors(v, p, k=2)
    nn_dist = nn_dist[:, 1]
    trim_dist_world = args.trim_distance * np.linalg.norm(voxel_size)  # Trim distance in world coordinates
    f_mask = np.stack([nn_dist[f[:, i]] < trim_dist_world for i in range(f.shape[1])], axis=-1)
    f_mask = np.all(f_mask, axis=-1)
    f = f[f_mask]

    print("Saving trimmed mesh...")
    pcu.save_mesh_vfn(args.out, v, f, n)
    print("Done!")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    mesh.vertex_normals = o3d.utility.Vector3dVector(-n)
    o3d.visualization.draw_geometries([pcd, mesh])


if __name__ == "__main__":
    main()
