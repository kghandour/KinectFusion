import os
import numpy as np
import math
import open3d as o3d
import time
import cv2 
from PIL import ImageOps
from layer import Layer
from tsdf import TSDFVolume

class Parser():
    def __init__(self,  sensor):
        self.sensor =  sensor
        self.fileBaseOut = "mesh_"
        depthIntrinsics =  self.sensor.m_depthIntrinsics
        depthIntrinsicsInv = np.linalg.inv(depthIntrinsics)
        depthExtrinsicsInv = np.linalg.inv( self.sensor.m_depthExtrinsics)
        self.tsdfVolume=TSDFVolume(depthIntrinsics)

        self.fX = depthIntrinsics[0, 0]
        self.fY = depthIntrinsics[1, 1]
        self.cX = depthIntrinsics[0, 2]
        self.cY = depthIntrinsics[1, 2]


    def visualize(self, vert_pos, vert_col):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vert_pos)
        pcd.colors = o3d.utility.Vector3dVector(vert_col/255)
        o3d.visualization.draw_geometries([pcd])

    # def compNormalsVerticesAndMask(self):

    #     ### Surface Measuremenet

    #     ## Dk  --------
    #     ##TODO: Make sure if Depth must be divided by 5000 or not. I normalized the depth directly to 255 instead of dividing by 5000
    #     imgConv = ImageOps.grayscale(self.sensor.dImageRaw)
    #     Dk = cv2.bilateralFilter(np.asarray(imgConv), 15,20,20)
    #     Dk = np.asarray(Dk)
    #     dHeight = self.sensor.m_depthImageHeight
    #     dWidth = self.sensor.m_depthImageWidth

    #     # Vk
    #     Vk = np.zeros((dHeight,  dWidth, 3))
    #     X, Y = np.meshgrid(dHeight, dWidth)
    #     for i in range( dHeight ):
    #         for j in range( dWidth ):
    #             x = (j - self.cX) / self.fX
    #             y = (i - self.cY) / self.fY
    #             depthAtPixel = Dk[i,j]
    #             if(depthAtPixel != 0):
    #                 Vk[i,j] = np.array([x*depthAtPixel, y*depthAtPixel, depthAtPixel])

        
    #     ## Nk
    #     Nk = np.zeros(( dHeight,  dWidth, 3))
    #     M = np.zeros((dHeight,dWidth))

    #     for u in range( dHeight -1): #Neighbouring
    #         for v in range( dWidth -1 ): #Neighbouring
    #             if 0 < Vk[u, v, 2] < 255:
    #                 n = np.cross( (Vk[u+1, v, :] - Vk[u,v,:]), Vk[u, v+1,:] - Vk[u,v,:])
    #                 if(np.linalg.norm(n)!=0 and Dk[u,v]!=0):
    #                     Nk[u, v, :] = n/np.linalg.norm(n)
    #                     M[u,v] = 1


    #     Vk = Vk[M==1]
    #     print(Vk.shape, Nk.shape, M.shape) ## Supposed to be HxW,3 , H,W,3 , H,W
    #     return Vk, Nk, M

    def one_loop(self):
        depthMap =  self.sensor.dImage
        colorMap =  self.sensor.rgbImage

        dHeight = self.sensor.m_depthImageHeight
        dWidth = self.sensor.m_depthImageWidth


        cameraSpace = np.zeros((dHeight,  dWidth, 4))

        trajectory =  self.sensor.currentTrajectory
        trajectoryInv = np.linalg.inv(trajectory)

        for i in range( self.sensor.m_depthImageHeight):
            for j in range( self.sensor.m_depthImageWidth):
                x = (j - self.cX) / self.fX
                y = (i - self.cY) / self.fY
                depthAtPixel = depthMap[i,j]
                if(depthAtPixel != -math.inf):
                    cameraSpace[i,j] = np.array([x*depthAtPixel, y*depthAtPixel, depthAtPixel, 1])
                else:
                    cameraSpace[i,j] = np.array([-math.inf, -math.inf, -math.inf, -math.inf])



        # vertices_position = np.zeros(( self.sensor.m_depthImageHeight,  self.sensor.m_depthImageWidth, 4))
        # vertices_color = np.full(( self.sensor.m_depthImageHeight,  self.sensor.m_depthImageWidth, 4), 255)
        vertices_position = []
        vertices_color = []
        for i in range( self.sensor.m_depthImageHeight):
            for j in range( self.sensor.m_depthImageWidth):
                depthAtPixel = depthMap[i,j]
                if(depthAtPixel != -math.inf):
                    vertices_position.append(trajectoryInv.dot(cameraSpace[i,j])[:3])
                    vertices_color.append([colorMap[i][j][0],colorMap[i][j][1],colorMap[i][j][2]])
                    # vertices_position[i,j] = trajectoryInv.dot(cameraSpace[i,j])
                #     vertices_color[i,j] = [colorMap[i][j][0],colorMap[i][j][1],colorMap[i][j][2],255]
                # else:
                #     vertices_position[i,j] = np.array([-math.inf, -math.inf, -math.inf, -math.inf])
                #     vertices_color[i,j] = np.array([0,0,0,0])


        fileName = os.path.join("mesh_out/",str(self.fileBaseOut)+str( self.sensor.currentIdx)+".off")
        # self.WriteMesh(vertices_position, vertices_color,  self.sensor.m_colorImageWidth,  self.sensor.m_colorImageHeight, fileName)
        # vert_pos, vert_col = self.cleanUp(vertices_position, vertices_color,  self.sensor.m_colorImageWidth,  self.sensor.m_colorImageHeight)
        return np.array(vertices_position), np.array(vertices_color)

    def process(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
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
            
            self.tsdfVolume.integrate(
                pyramid["l1"].depthImage, pyramid["l1"].rgbImage, np.eye(4))

            exit()
            st = time.time()
            vert_pos, vert_col = self.one_loop()
            et = time.time()
            print("Time taken to process a frame ", et - st)
            pcd.points = o3d.utility.Vector3dVector(vert_pos)
            pcd.colors = o3d.utility.Vector3dVector(vert_col/255)
            if(i==0):
                vis.add_geometry(pcd)
            else:
                vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            i+=1
        vis.destroy_window()


        