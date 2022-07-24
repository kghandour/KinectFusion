import cv2
from cv2 import destroyWindow
import numpy as np
from PIL import ImageOps, Image
from camera_sensors import CamDetails

from transforms import Transforms


class Layer():
    def __init__(self,  depthImage, rgbImage, sensor):
        self.sensor = sensor
        self.depthImage = depthImage
        self.rgbImage = np.swapaxes(rgbImage, 0,1)
        self.dHeight = self.depthImage.size[1]
        self.dWidth = self.depthImage.size[0]


        depthIntrinsics =  self.sensor.m_depthIntrinsics
        self.fX = depthIntrinsics[0, 0]
        self.fY = depthIntrinsics[1, 1]
        self.cX = depthIntrinsics[0, 2]
        self.cY = depthIntrinsics[1, 2]


        self.Dk = []
        self.Vk = np.zeros((self.dWidth,  self.dHeight, 3))
        self.Nk = np.zeros(( self.dWidth,  self.dHeight, 3))
        self.M = np.zeros((self.dWidth, self.dHeight))

        self.compNormalsVerticesAndMask()


    def compNormalsVerticesAndMask(self):
        ### Surface Measuremenet

        ## Dk  --------
        ##TODO: Make sure if Depth must be divided by 5000 or not. I normalized the depth directly to 255 instead of dividing by 5000
        normalize = np.array(self.depthImage)/72
        image = Image.fromarray(normalize)
        imgConv = ImageOps.grayscale(image)
        Dk = cv2.bilateralFilter(np.asarray(imgConv), 15,20,20)
        self.Dk = np.transpose(np.asarray(Dk))
        # Vk = Camera space

        self.Vk = Transforms.screen2cam(self.Dk)
        # X_range = range(self.dWidth)
        # Y_range = range(self.dHeight)
        # X, Y = np.meshgrid(X_range, Y_range)
        # Z = (Dk / 5000).reshape(-1, 1)
        # X = (X.reshape(-1, 1) - CamDetails.cX) * Z / CamDetails.fX
        # Y = (Y.reshape(-1, 1) - CamDetails.cY) * Z / CamDetails.fY
        # self.Vk = np.hstack([X, Y, Z]).reshape(dHeight, dWidth, 3)
        # for i in range( dHeight ):
        #     for j in range( dWidth ):
        #         x = (j - self.cX) / self.fX
        #         y = (i - self.cY) / self.fY
        #         depthAtPixel = Dk[i,j]
        #         if(depthAtPixel != 0):
        #             self.Vk[i,j] = np.array([x*depthAtPixel, y*depthAtPixel, depthAtPixel])
        #             self.Vk_h[i,j] = np.array([x*depthAtPixel, y*depthAtPixel, depthAtPixel,1])

        # Nk
        # M
        for u in range( self.dWidth -1): #Neighbouring
            for v in range( self.dHeight -1 ): #Neighbouring
                n = np.zeros(3, )
                if 1 <= self.Vk[u, v, 2] <= 240:
                    n = np.cross((self.Vk[u+1, v,:] - self.Vk[u,v,:]), self.Vk[u, v+1,:] - self.Vk[u,v,:])
                if(np.linalg.norm(n)!=0):
                    self.Nk[u, v, :] = n/np.linalg.norm(n)
                    self.M[u,v] = 1

        ## Supposed to be HxW,3 , H,W,3 , H,W
        self.Vk = self.Vk[self.M == 1].reshape(-1,3)
        self.Nk = self.Nk[self.M == 1].reshape(-1,3)
        self.rgbImage = self.rgbImage[self.M == 1].reshape(-1,3)
