
import torch
import numpy as np
import math
import open3d as o3d
from config import config
import time

from camera_sensors import CamDetails
class Transforms():
    @staticmethod
    def world2cam(world_matrix, pose):
        extrinsic_inv = np.linalg.inv(pose)
        volume_camera_cordinates = torch.matmul(world_matrix, extrinsic_inv.T)  ##
        return volume_camera_cordinates

    @staticmethod
    def cam2world(cameraSpace, pose):
        trajectory = pose
        # worldSpace = torch.matmul(cameraSpace, pose.T)  ##
        worldSpace = cameraSpace @ trajectory[:3, :3].T + trajectory[:3, 3].reshape(1,3)
        return worldSpace

    @staticmethod
    def screen2cam(depthMap, vis_3d=False):
        cameraSpace = torch.ones((depthMap.shape[0], depthMap.shape[1],3))
        cameraSpace.to(config.getTorchDevice())
        # intermediate = torch.matmul(
        #     torch.from_numpy(depthMap[:, :2]), cameraSpace).long()
        # cameraSpace = torch.matmul(
        #     intermediate[:, :3], np.linalg.inv(CamDetails.depthIntrinsics).T)
        
        
        st = time.time()
        ## Using torch
        X_range = torch.arange(0,depthMap.shape[0])
        Y_range = torch.arange(0,depthMap.shape[1])
        X,Y  = torch.meshgrid((X_range, Y_range), indexing='ij')
        

        Z = (depthMap).reshape(-1, 1)

        X = (X.reshape(-1, 1) - CamDetails.cX) * Z / CamDetails.fX
        Y = (Y.reshape(-1, 1) - CamDetails.cY) * Z / CamDetails.fY
        cameraSpace = torch.hstack([X, Y, Z]).reshape(depthMap.shape[0], depthMap.shape[1], 3)



        
        ## Using loops
        # for i in range(depthMap.shape[0]):
        #     for j in range(depthMap.shape[1]):
        #         x = (i - CamDetails.cX) / CamDetails.fX
        #         y = (j - CamDetails.cY) / CamDetails.fY
        #         depthAtPixel = depthMap[i,j]
        #         if(depthAtPixel != -math.inf):
        #             cameraSpace[i,j] = torch.tensor([x*depthAtPixel, y*depthAtPixel, depthAtPixel])



        if(vis_3d):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cameraSpace.reshape(-1,3))
            o3d.visualization.draw_geometries([pcd])
            exit()
        return cameraSpace

    @staticmethod
    def cam2screen(pcd):
        imageSpace = np.zeros((CamDetails.depthHeight, CamDetails.depthWidth))
        intermediate_array = torch.matmul(
            pcd.positions[:, :3], CamDetails.depthIntrinsics.T)
        imageSpace = torch.divide(
            intermediate_array[:, :2], intermediate_array, rounding_mode='floor').long()
        # for i in range(pcd.points):
        #     x,y,z = pcd.points[i]
        #     x = x*CamDetails.fX/z + CamDetails.cX
        #     y = y*CamDetails.fY/z + CamDetails.cY
        #     imageSpace[x,y] = pcd.colors[x,y,z]*255
        return imageSpace
        
    

    


