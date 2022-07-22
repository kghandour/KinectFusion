import os
import numpy as np
import math
import open3d as o3d
import time
import cv2
from PIL import ImageOps
from camera_sensors import CamDetails
from icp import ICPOptimizer
from raycast import Raycast
from layer import Layer
from transforms import Transforms
from tsdf import TSDFVolume


class Parser():
    def __init__(self,  sensor):
        self.sensor = sensor
        self.fileBaseOut = "mesh_"
        self.tsdfVolume=TSDFVolume(CamDetails.depthIntrinsics)
        self.pyramids_so_far = []
        self.Transformation_loc2glo_list = []
        self.T_matrix = np.eye(4)
        self.icp_optimizer = ICPOptimizer()

    def visualize(self, vert_pos, vert_col):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vert_pos)
        pcd.colors = o3d.utility.Vector3dVector(vert_col/255)
        o3d.visualization.draw_geometries([pcd])

    def one_loop(self):
        depthMap = self.sensor.dImage
        colorMap = self.sensor.rgbImage

        dHeight = self.sensor.m_depthImageHeight
        dWidth = self.sensor.m_depthImageWidth

        cameraSpace = np.zeros((dHeight,  dWidth, 4))

        trajectory = self.sensor.currentTrajectory
        trajectoryInv = np.linalg.inv(trajectory)

        for i in range(self.sensor.m_depthImageHeight):
            for j in range(self.sensor.m_depthImageWidth):
                x = (j - CamDetails.cX) / CamDetails.fX
                y = (i - CamDetails.cY) / CamDetails.fY
                depthAtPixel = depthMap[i, j]
                if(depthAtPixel != -math.inf):
                    cameraSpace[i, j] = np.array(
                        [x*depthAtPixel, y*depthAtPixel, depthAtPixel, 1])
                else:
                    cameraSpace[i, j] = np.array(
                        [-math.inf, -math.inf, -math.inf, -math.inf])

        # vertices_position = np.zeros(( self.sensor.m_depthImageHeight,  self.sensor.m_depthImageWidth, 4))
        # vertices_color = np.full(( self.sensor.m_depthImageHeight,  self.sensor.m_depthImageWidth, 4), 255)
        vertices_position = []
        vertices_color = []
        for i in range(self.sensor.m_depthImageHeight):
            for j in range(self.sensor.m_depthImageWidth):
                depthAtPixel = depthMap[i, j]
                if(depthAtPixel != -math.inf):
                    vertices_position.append(
                        trajectoryInv.dot(cameraSpace[i, j])[:3])
                    vertices_color.append(
                        [colorMap[i][j][0], colorMap[i][j][1], colorMap[i][j][2]])
                    # vertices_position[i,j] = trajectoryInv.dot(cameraSpace[i,j])
                #     vertices_color[i,j] = [colorMap[i][j][0],colorMap[i][j][1],colorMap[i][j][2],255]
                # else:
                #     vertices_position[i,j] = np.array([-math.inf, -math.inf, -math.inf, -math.inf])
                #     vertices_color[i,j] = np.array([0,0,0,0])

        fileName = os.path.join(
            "mesh_out/", str(self.fileBaseOut)+str(self.sensor.currentIdx)+".off")
        # self.WriteMesh(vertices_position, vertices_color,  self.sensor.m_colorImageWidth,  self.sensor.m_colorImageHeight, fileName)
        # vert_pos, vert_col = self.cleanUp(vertices_position, vertices_color,  self.sensor.m_colorImageWidth,  self.sensor.m_colorImageHeight)
        return np.array(vertices_position), np.array(vertices_color)

    def process(self):
        vis_pcd = o3d.visualization.Visualizer()
        vis_pcd.create_window()
        vis_volume = o3d.visualization.Visualizer()
        vis_volume.create_window()

        i = 0
        pcd = o3d.geometry.PointCloud()
        while( self.sensor.processNextFrame()):
            depthImageRaw =  self.sensor.dImage
            colorImageRaw =  self.sensor.rgbImage
            h, w = depthImageRaw.shape
            pyramid = {}
            pyramid['l1'] = Layer(depthImageRaw, colorImageRaw, self.sensor)
            pyramid['l2'] = Layer(cv2.resize(depthImageRaw, (int(w/2), int(h/2))), cv2.resize(colorImageRaw, (int(w/2), int(h/2))), self.sensor)
            pyramid['l3'] = Layer(cv2.resize(depthImageRaw, (int(w/4), int(h/4))), cv2.resize(colorImageRaw, (int(w/4), int(h/4))), self.sensor)
            self.pyramids_so_far.append(pyramid)
            if(i==0):
                curr_volume = self.tsdfVolume.integrate(pyramid["l1"].depthImage, pyramid["l1"].rgbImage,self.T_matrix, None)
            else:
                self.T_matrix = self.icp_optimizer.estimate_pose(pyramid["l1"].Vk,self.pyramids_so_far[-1]["l1"].Vk,pyramid["l1"].Nk,self.pyramids_so_far[-1]["l1"].Nk)
                curr_volume = self.tsdfVolume.integrate(pyramid["l1"].depthImage, pyramid["l1"].rgbImage,self.T_matrix, self.prev_volume)    
                self.Transformation_list.append(self.T_matrix)
            
            tsdf_volume_mesh = self.tsdfVolume.visualize()
            self.prev_volume = curr_volume

            
            # exit()
            st = time.time()
            vert_pos, vert_col = self.one_loop()
            et = time.time()
            print("Time taken to process a frame ", et - st)
            pcd.points = o3d.utility.Vector3dVector(vert_pos)
            pcd.colors = o3d.utility.Vector3dVector(vert_col/255)
            if(i == 0):
                vis_pcd.add_geometry(pcd)
                vis_volume.add_geometry(tsdf_volume_mesh)
            else:
                vis_pcd.update_geometry(pcd)
                vis_volume.update_geometry(tsdf_volume_mesh)

            vis_pcd.poll_events()
            vis_pcd.update_renderer()
            vis_volume.poll_events()
            vis_volume.update_renderer()
            i += 1
        vis_pcd.destroy_window()
        vis_volume.destroy_window()


        
