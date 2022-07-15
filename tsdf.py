from pickle import TRUE
from sys import exc_info
import time
import numpy as np
import torch
import copy

from dataset import Dataset
from parser import Parser
from virtualSensor import VirtualSensor


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

    def __init__(self, min_bounds, max_bounds, voxel_size, camera_intrinsics, margin=5):
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

            self.volume_voxel_indices = np.indices(
                self.volume_size).reshape(3, -1).T

            volume_world_cordinates = self.min_bounds + \
                (self.voxel_size*self.volume_voxel_indices)
            self.volume_world_cordinates = torch.cat(
                (volume_world_cordinates, torch.ones(volume_world_cordinates.shape[0], 1)), dim=-1)

    def integrate(self, depthImage, rgbImage, pose_estimation):
        height, width = depthImage.shape
        pose = torch.from_numpy(pose_estimation)
        depth = torch.from_numpy(rgbImage)
        extrinsic_inv = torch.inverse(pose)

        # Convert volume world cordinates to camera cordinates
        volume_camera_cordinates = torch.matmul(
            self.volume_world_cordinates, extrinsic_inv.T)

        # RK
        # Z values of camera cordinates
        volume_camera_cordinates_z = torch.broadcast_to(
            volume_camera_cordinates[:, 2], (2, -1)).T
        # Convert volume camera cordinates to pixel cordinates
        volume_pixel_cordinates_xyz = torch.matmul(
            volume_camera_cordinates[:, :3], self.camera_intrinsics_T)

        # divide by z and remove z
        volume_pixel_cordinates_xy = torch.divide(
            volume_pixel_cordinates_xyz[:, :2], volume_camera_cordinates_z, rounding_mode='trunc')

        # get indices of valid pixels
        volume_camera_valid_pixels = torch.where((volume_pixel_cordinates_xy[:, 0] >= 0) &
                                                 (volume_pixel_cordinates_xy[:, 1] >= 0) &
                                                 (volume_pixel_cordinates_xy[:, 0] < width) &
                                                 (volume_pixel_cordinates_xy[:, 1] < height) &
                                                 (volume_camera_cordinates_z[:, 0] > 0),
                                                 True, False)
        # Apply indexing  to get the depth value of valid pixels
        volume_camera_cordinates_depth_used = volume_camera_cordinates_z[:,
                                                                         0][volume_camera_valid_pixels]
        # Get the valid depth valuse of the source img corrosponding to valid projections from the volume
        depth_img_used = depth[volume_pixel_cordinates_xy[volume_camera_valid_pixels].split(
            1, 1)].reshape(-1,)
        # distance from the camera center along each depthmap ray,
        r = depth_img_used-volume_camera_cordinates_depth_used

        # valid depths that lay inside
        valid_depth_img_points = torch.where(
            (depth_img_used > 0) & r > self.trunc_distance)

        # Integrate TSDF

        # voxel cordinates that correspond to the valid depth img cordinates
        volume_voxel_cordinates_used = self.vox_coords[valid_depth_img_points]

        # The old weights before integration
        old_volume_weight_values_used = copy.deepcopy(
            self._weight_vol_cpu[valid_depth_img_points])
        # the old tsdf before integration
        old_volume_tsdf_values_used = copy.deepcopy(
            self._tsdf_vol_cpu[valid_depth_img_points])

        # clamp far distances
        dist = torch.min(1, r / self.trunc_distance)[valid_depth_img_points]

        # Fk(p)  = ( ( Wk−1(p) * Fk−1(p) ) + ( WRk(p) * FRk(p) ) ) / ( Wk−1(p) + WRk(p))
        self.tsdf_volume[volume_voxel_cordinates_used] = (
            (old_volume_tsdf_values_used*old_volume_weight_values_used)+(dist*1))/(old_volume_weight_values_used+1)
        # Wk(p)  =Wk−1(p)+WRk(p)
        self.weight_volume[volume_voxel_cordinates_used] = torch.add(
            1, old_volume_weight_values_used)


if __name__ == '__main__':
    pass
    # dataset = Dataset("rgbd_dataset_freiburg1_xyz")
    # sensor = VirtualSensor(dataset, 800)
    # parser = Parser(sensor)
    # parser.process()

    # maxX = 2
    # maxY = 2.2
    # maxZ = 3

    # minX = -0.5
    # minY = -0.8
    # minZ = -1

    # min_bounds = np.asarray([minX, minY, minZ])
    # max_bounds = np.asarray([maxX, maxY, maxZ])

    # voxel_size = 0.03

    # volume_size = np.ceil(
    #     (max_bounds-min_bounds)/voxel_size).astype(int)

    # TSDFVolume(min_bounds=min_bounds,
    #            max_bounds=max_bounds, voxel_size=voxel_size)
