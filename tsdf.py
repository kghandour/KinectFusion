from pickle import TRUE
from sys import exc_info
import time
import numpy as np
import torch


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

    def __init__(self, min_bounds, max_bounds, voxel_size, camera_intrinsics, margin=2):
        # torch.backends.cuda.matmul.allow_tf32 = Falses
        self.min_bounds = np.asarray(min_bounds)
        self.max_bounds = np.asarray(max_bounds)
        self.voxel_size = float(voxel_size)
        self.camera_intrinsics = torch.from_numpy(camera_intrinsics)
        self.camera_intrinsics_T = self.camera_intrinsics.T
        self.trunc_distance = self.volume_size*margin

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

        volume_camera_cordinates = torch.matmul(
            self.volume_world_cordinates, extrinsic_inv.T)

        # RK
        volume_camera_cordinates_z = torch.broadcast_to(
            volume_camera_cordinates[:, 2], (2, -1)).T
        volume_pixel_cordinates_xyz = torch.matmul(
            volume_camera_cordinates[:, :3], self.camera_intrinsics_T)
        volume_pixel_cordinates_xy = torch.divide(
            volume_pixel_cordinates_xyz[:, :2], volume_camera_cordinates_z, rounding_mode='trunc')

        volume_camera_valid_pixels = torch.where((volume_pixel_cordinates_xy[:, 0] >= 0) &
                                                 (volume_pixel_cordinates_xy[:, 1] >= 0) &
                                                 (volume_pixel_cordinates_xy[:, 0] < width) &
                                                 (volume_pixel_cordinates_xy[:, 1] < height) &
                                                 (volume_camera_cordinates_z[:, 0] > 0),
                                                 True, False)
        volume_camera_cordinates_depth_used = volume_camera_cordinates_z[:,
                                                                         0][volume_camera_valid_pixels]
        depth_img_used = depthImage[volume_pixel_cordinates_xy[volume_camera_valid_pixels].split(
            1, 1)].reshape(-1,)

        r = depth_img_used-volume_camera_cordinates_depth_used


if __name__ == '__main__':

    # depth_im = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [
    #                         11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]])
    # pix_y = torch.tensor([0, 1, 2, 3, 4, 5, 2, 0])
    # pix_x = torch.tensor([2, 4, 4, 5, 8, 3, 2, 4])

    # pixy_xy = torch.tensor([[0, 2], [1, 4], [2, 4], [3, 5], [
    #                        4, 8], [5, 3], [2, 2], [0, 4]])
    # valid_pix = torch.tensor(
    #     [True, True, True, False, False, False, True, False])
    # s1 = time.time()
    # print(depth_im[pix_y[valid_pix], pix_x[valid_pix]])
    # print(time.time()-s1)
    # s2 = time.time()
    # print((depth_im[pixy_xy[valid_pix].split(1, 1)].reshape(-1, )))
    # print(time.time()-s2)

    # print(pixy_xy[valid_pix].split(1, 1))
    # print(pixy_xy[valid_pix])
    # exit()
    print('')

    maxX = 2
    maxY = 2.2
    maxZ = 3

    minX = -0.5
    minY = -0.8
    minZ = -1

    min_bounds = np.asarray([minX, minY, minZ])
    max_bounds = np.asarray([maxX, maxY, maxZ])

    voxel_size = 0.03

    volume_size = np.ceil(
        (max_bounds-min_bounds)/voxel_size).astype(int)

    # TSDFVolume(min_bounds=min_bounds,
    #            max_bounds=max_bounds, voxel_size=voxel_size)

    # volume_voxel_indices = np.indices(volume_size).reshape(3, -1).T

    # volume_world_cordinates = torch.tensor(min_bounds +
    #                                        (voxel_size*volume_voxel_indices))

    # volume_world_cordinates = torch.cat(
    #     (volume_world_cordinates, torch.ones(volume_world_cordinates.shape[0], 1)), dim=-1)

    # # print(volume_voxel_indices[:5])

    # # print((volume_voxel_indices[:5])[[False, True, False, True, False], 2])
    # # print(volume_size)

    # pose_estimation = torch.from_numpy(
    #     np.array([[1, 0, 0, 2], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=np.float64))

    # extrinsic_inv = torch.inverse(pose_estimation)
    # # # print(extrinsic_inv)

    # cam_c_1 = torch.matmul(volume_world_cordinates, extrinsic_inv.T
    #                        )

    # m_colorIntrinsics = torch.from_numpy(
    #     np.array([525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1]).reshape((3, 3)))

    # # print(m_colorIntrinsics)

    # # print(cam_c_1[0])
    # # print(cam_c_1[:, :3].shape)

    # volume_pixel_cordinates = (torch.matmul(
    #     cam_c_1[:, :3], m_colorIntrinsics.T))

    # cam_z = torch.broadcast_to(cam_c_1[:, 2], (2, -1)).T

    # # print(volume_pixel_cordinates.shape)
    # pix_xy = (torch.divide(
    #     volume_pixel_cordinates[:, :2], cam_z, rounding_mode='trunc'))

    # z_5 = cam_z[:5, :]
    # pix_xy_5 = pix_xy[:5, :]

    # # print(pix_xy_5)

    # z_test = torch.tensor([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]])
    # print(pix_xy_5)
    # mask = (torch.where((pix_xy_5[:, 0] >= 0) & (pix_xy_5[:, 1] >= 0) & (pix_xy_5[:, 0] % 2 == 0) & (pix_xy_5[:, 1] % 1 == 0) &
    #                     (z_test[:, 0] == 1), True, False))
    # print(mask)
    # print(z_5[:, 0])
    # print(z_5[:, 0][mask])
    # print(volume_pixel_cordinates.shape)

    # print(volume_pixel_cordinates[0])
    # print(volume_pixel_cordinates.shape)
    # print(torch.equal(cam_c_1, cam_c_2))

    # source_points.dot(pose_estimation.T)
