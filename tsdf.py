from tokenize import Double
from turtle import width
import numpy as np
import torch
import copy
from skimage import measure
import open3d as o3d

from config import config


class TSDFVolume:
    """Constructor.
    Args:
        min_bounds (ndarray): An ndarray of shape (3,). Specifies the
            minimum xyz bounds.
        max_bounds (ndarray): An ndarray of shape (3,). Specifies the
            maximum xyz bounds.
        num_voxels (ndarray): An ndarray of shape (3,). Specifies the
            number of voxels in the xyz directirons.
    """
    # Bounds, lower x,z,y [ right(flipped from original), deep, ]

    def __init__(self, camera_intrinsics, min_bounds=[-3.5, -3.5, -3.5], max_bounds=[3.5, 3.5, 3.5], voxel_size=0.03, margin=3):
        # torch.backends.cuda.matmul.allow_tf32 = Falses
        self.min_bounds = np.asarray(min_bounds)
        self.max_bounds = np.asarray(max_bounds)
        self.voxel_size = float(voxel_size)
        self.camera_intrinsics = torch.from_numpy(camera_intrinsics)
        self.camera_intrinsics_T = self.camera_intrinsics.T
        self.trunc_distance = self.voxel_size*margin

        self.volume_size = np.ceil(
            (self.max_bounds-self.min_bounds)/self.voxel_size).astype(int)
        

        with torch.no_grad():
            self.tsdf_volume = torch.ones(
                tuple(self.volume_size), dtype=torch.float64)
            self.weight_volume = torch.zeros(
                tuple(self.volume_size), dtype=torch.float32)
            self.rgb_volume = torch.zeros(
                (self.volume_size[0], self.volume_size[1], self.volume_size[2], 3), dtype=torch.float32)

            self.volume_voxel_indices = np.indices(
                self.volume_size).reshape(3, -1).T
            volume_world_cordinates = self.min_bounds + \
                (self.voxel_size*self.volume_voxel_indices)
            self.volume_world_cordinates = torch.cat(
                (torch.from_numpy(volume_world_cordinates), torch.ones(volume_world_cordinates.shape[0], 1)), dim=-1)
            self.volume_voxel_indices = torch.from_numpy(
                self.volume_voxel_indices)

    def integrate(self, depthImage, rgbImage, pose_estimation, weight=1):
        with torch.no_grad():
            depthImage = np.swapaxes(np.array(depthImage), 0, 1)
            rgbImage = np.swapaxes(rgbImage, 0, 1)
            height, width, _ = depthImage.shape
            pose = torch.from_numpy(pose_estimation)
            depth = torch.from_numpy(depthImage)

            if(not config.useGroundTruth()):
                extrinsic_inv = torch.inverse(pose)
                volume_camera_cordinates = torch.matmul(
                    self.volume_world_cordinates, extrinsic_inv)
            else:
                volume_camera_cordinates = torch.matmul(
                    self.volume_world_cordinates, pose)

            # Convert volume world cordinates to camera cordinates
            # volume_camera_cordinates = torch.matmul(
            #     self.volume_world_cordinates, extrinsic_inv)

            # RK
            # Z values of camera cordinates
            volume_camera_cordinates_z = torch.broadcast_to(
                volume_camera_cordinates[:, 2], (2, -1)).T
            # print(torch.min(volume_camera_cordinates_z[:, 0]), torch.max(volume_camera_cordinates_z[:, 0]))

            # Convert volume camera cordinates to pixel cordinates
            volume_pixel_cordinates_xyz = torch.matmul(
                volume_camera_cordinates[:, :3], self.camera_intrinsics_T)

            # divide by z and remove z
            volume_pixel_cordinates_xy = torch.divide(
                volume_pixel_cordinates_xyz[:, :2], volume_camera_cordinates_z, rounding_mode='floor').long()
            # get indices of valid pixels

            volume_pixel_cordinates_xy = torch.nan_to_num(
                volume_pixel_cordinates_xy, nan=-1)
            volume_camera_cordinates_z = torch.nan_to_num(
                volume_camera_cordinates_z, nan=-10)
            volume_camera_valid_pixels = torch.where((volume_pixel_cordinates_xy[:, 0] >= 0) &
                                                     (volume_pixel_cordinates_xy[:, 1] >= 0) &
                                                     (volume_pixel_cordinates_xy[:, 0] < height) &
                                                     (volume_pixel_cordinates_xy[:, 1] < width) &
                                                     (volume_camera_cordinates_z[:, 0] > -2), True, False
                                                     )

            # Apply indexing  to get the depth value of valid pixels
            volume_camera_cordinates_depth_used = volume_camera_cordinates_z[:,
                                                                             0][volume_camera_valid_pixels]
            # Get the valid depth values of the source img corrosponding to valid projections from the volume
            depth_img_used = depth[volume_pixel_cordinates_xy[volume_camera_valid_pixels].split(
                1, 1)].reshape(-1,)

            # distance from the camera center along each depthmap ray,
            r = depth_img_used-volume_camera_cordinates_depth_used

            # valid depths that lay inside
            valid_depth_img_points = torch.where(
                (depth_img_used > 0) & (r >= -self.trunc_distance), True, False)

            # voxel cordinates that correspond to the valid depth img cordinates
            volume_voxel_cordinates_used = (
                self.volume_voxel_indices[volume_camera_valid_pixels])[valid_depth_img_points].long()

            cordinates = volume_voxel_cordinates_used[:,
                                                      0], volume_voxel_cordinates_used[:, 1], volume_voxel_cordinates_used[:, 2]

            # The old weights before integration
            old_volume_weight_values_used = copy.deepcopy(
                self.weight_volume[cordinates])
            # the old tsdf before integration
            old_volume_tsdf_values_used = copy.deepcopy(
                self.tsdf_volume[cordinates])

            # clamp far distances
            dist = torch.min(torch.tensor(1), r /
                             self.trunc_distance)[valid_depth_img_points]

            weight_tensor = torch.full(dist.shape, weight)

            # Fk(p)  = ( ( Wk−1(p) * Fk−1(p) ) + ( WRk(p) * FRk(p) ) ) / ( Wk−1(p) + WRk(p))
            self.tsdf_volume[cordinates] = (
                (old_volume_tsdf_values_used*old_volume_weight_values_used)+(dist*weight_tensor))/(old_volume_weight_values_used+weight_tensor)
            # Wk(p)  =Wk−1(p)+WRk(p)
            self.weight_volume[cordinates] = torch.add(
                weight_tensor, old_volume_weight_values_used)

            # RGB values of the depth used
            rgb_used = rgbImage[volume_pixel_cordinates_xy[volume_camera_valid_pixels].split(
                1, 1)].reshape(-1, 3)[valid_depth_img_points]
            # the old colors before integration
            old_volume_rgb_values_used = copy.deepcopy(
                self.rgb_volume[cordinates])
            # [:,none] => to make weight as vector not float to be able to multiplicate ([1,2,3] =>[[1],[2],[3]])
            self.rgb_volume[cordinates] = ((old_volume_weight_values_used[:, None] * old_volume_rgb_values_used) + (weight_tensor[:, None] * rgb_used)) / (
                (old_volume_weight_values_used+weight_tensor)[:, None])

    def visualize(self):
        vertices, triangles, vertex_normals, vals = measure.marching_cubes(
            self.tsdf_volume.cpu().numpy(), level=None)
        volume_cordinates = np.round(vertices).astype(int)
        # volume indices to world coordinates
        vertices = vertices * self.voxel_size + self.min_bounds

        vertex_colors = self.rgb_volume[volume_cordinates[:, 0],
                                        volume_cordinates[:, 1], volume_cordinates[:, 2]].cpu().numpy()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(float))
        mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            vertex_colors.astype(np.uint8) / 255.)

        mesh.transform(np.array([
            [1, 0, 0, 0.5], [0, -1, 0, 0.45], [0, 0, 1, -0.1], [0, 0, 0, 1]

        ]))
        mesh.rotate(mesh.get_rotation_matrix_from_xyz(
            (0, 0, np.pi / 2)), center=(0, 0, 0))

        mesh.compute_vertex_normals()

        normals = mesh.vertex_normals

        return mesh, normals
