import numpy as np
import torch
import copy

from volume import Volume


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

    def __init__(self, camera_intrinsics, min_bounds=[-2.5, -2, -0.2], max_bounds=[1.5, 2.5, 1.5], voxel_size=0.04, margin=3, origin=(0,0,0)):
        # torch.backends.cuda.matmul.allow_tf32 = Falses
        self.origin = origin
        self.min_bounds = np.asarray(min_bounds)
        self.max_bounds = np.asarray(max_bounds)
        self.voxel_size = float(voxel_size)
        self.camera_intrinsics = torch.from_numpy(camera_intrinsics)
        self.camera_intrinsics_T = self.camera_intrinsics.T
        self.trunc_distance = self.voxel_size*margin

        self.volume_size = np.ceil(
            (self.max_bounds-self.min_bounds)/self.voxel_size).astype(int)

        with torch.no_grad():
            self.tsdf_volume = torch.full(
                tuple(self.volume_size),-1, dtype=torch.float64)
            self.weight_volume = torch.zeros(
                tuple(self.volume_size), dtype=torch.float32)

            self.volume_voxel_indices = np.indices(
                self.volume_size).reshape(3, -1).T

            volume_world_cordinates = self.min_bounds + \
                (self.voxel_size*self.volume_voxel_indices)
            self.volume_world_cordinates = torch.cat(
                (torch.from_numpy(volume_world_cordinates), torch.ones(volume_world_cordinates.shape[0], 1)), dim=-1)
            self.volume_voxel_indices = torch.from_numpy(
                self.volume_voxel_indices)

    def Volume2World(self, x, y, z):
        worldSpace = np.zeros(3)
        worldSpace[0] = x * self.voxel_size
        worldSpace[1] = y * self.voxel_size
        worldSpace[2] = z * self.voxel_size
        worldSpace += self.origin
        return worldSpace

    def integrate(self, depthImage, rgbImage, pose_estimation, prev_volume: Volume, weight=1):
        with torch.no_grad():
            height, width = depthImage.shape
            pose = torch.from_numpy(pose_estimation)
            depth = torch.from_numpy(depthImage)
            extrinsic_inv = torch.inverse(pose)

            # Convert volume world cordinates to camera cordinates
            volume_camera_cordinates = torch.matmul(
                self.volume_world_cordinates, extrinsic_inv.T)  ## Lambda 

            # RK
            # Z values of camera cordinates
            volume_camera_cordinates_z = torch.broadcast_to(
                volume_camera_cordinates[:, 2], (2, -1)).T
            # Convert volume camera cordinates to pixel cordinates
            volume_pixel_cordinates_xyz = torch.matmul(
                volume_camera_cordinates[:, :3], self.camera_intrinsics_T)

            # divide by z and remove z
            volume_pixel_cordinates_xy = torch.divide(
                volume_pixel_cordinates_xyz[:, :2], volume_camera_cordinates_z, rounding_mode='floor').long()
            # get indices of valid pixels

            volume_camera_valid_pixels = torch.where((volume_pixel_cordinates_xy[:, 0] >= 0) &
                                                     (volume_pixel_cordinates_xy[:, 1] >= 0) &
                                                     (volume_pixel_cordinates_xy[:, 0] < height) &
                                                     (volume_pixel_cordinates_xy[:, 1] < width) &
                                                     (volume_camera_cordinates_z[:, 0] > 0), True, False
                                                     )


            # Apply indexing  to get the depth value of valid pixels
            volume_camera_cordinates_depth_used = volume_camera_cordinates_z[:,
                                                                             0][volume_camera_valid_pixels]

                                                                ## TODO: NANs 

            # Get the valid depth valuse of the source img corrosponding to valid projections from the volume
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
            if (prev_volume is not None):
                old_volume_weight_values_used = prev_volume.weights_volume
                # the old tsdf before integration
                old_volume_tsdf_values_used = prev_volume.tsdf_volume

            # clamp far distances
            dist = torch.min(torch.tensor(1), r /
                             self.trunc_distance)[valid_depth_img_points]


            weight_tensor = torch.full(dist.shape, weight)
            # Fk(p)  = ( ( Wk−1(p) * Fk−1(p) ) + ( WRk(p) * FRk(p) ) ) / ( Wk−1(p) + WRk(p))
            if (prev_volume is not None):
                self.tsdf_volume[cordinates] = (
                    (old_volume_tsdf_values_used*old_volume_weight_values_used)+(dist*weight_tensor))/(old_volume_weight_values_used+weight_tensor)
                # Wk(p)  =Wk−1(p)+WRk(p)
                self.weight_volume[cordinates] = torch.add(
                    weight_tensor, old_volume_weight_values_used)
            else:
                self.tsdf_volume[cordinates] = ((dist*weight_tensor))
                self.weight_volume[cordinates] = weight_tensor

            curr_vol = Volume(self.tsdf_volume, self.weight_volume)
            return curr_vol

