import cv2
from cv2 import destroyWindow
import numpy as np
from PIL import ImageOps

from transforms import Transforms


class Layer():
    def __init__(self,  depthImage, rgbImage, sensor):
        self.sensor = sensor
        self.depthImage = depthImage
        self.rgbImage = rgbImage
        self.dHeight = self.depthImage.shape[0]
        self.dWidth = self.depthImage.shape[1]


        depthIntrinsics =  self.sensor.m_depthIntrinsics
        self.fX = depthIntrinsics[0, 0]
        self.fY = depthIntrinsics[1, 1]
        self.cX = depthIntrinsics[0, 2]
        self.cY = depthIntrinsics[1, 2]


        self.Dk = []
        self.Vk = np.zeros((self.dHeight,  self.dWidth, 3))
        self.Vk_h = np.zeros((self.dHeight,  self.dWidth, 4))
        self.Nk = np.zeros(( self.dHeight,  self.dWidth, 3))
        self.M = np.zeros((self.dHeight, self.dWidth))

        self.compNormalsVerticesAndMask()


    def compNormalsVerticesAndMask(self):
        ### Surface Measuremenet

        ## Dk  --------
        ##TODO: Make sure if Depth must be divided by 5000 or not. I normalized the depth directly to 255 instead of dividing by 5000
        imgConv = ImageOps.grayscale(self.sensor.dImageRaw)
        Dk = cv2.bilateralFilter(np.asarray(imgConv), 15,20,20)
        Dk = np.asarray(Dk)


        dHeight = self.dHeight
        dWidth = self.dWidth

        # Vk = Camera space
        
        X, Y = np.meshgrid(dHeight, dWidth)
        self.Vk = Transforms.screen2cam(Dk)
        # for i in range( dHeight ):
        #     for j in range( dWidth ):
        #         x = (j - self.cX) / self.fX
        #         y = (i - self.cY) / self.fY
        #         depthAtPixel = Dk[i,j]
        #         if(depthAtPixel != 0):
        #             self.Vk[i,j] = np.array([x*depthAtPixel, y*depthAtPixel, depthAtPixel])
        #             self.Vk_h[i,j] = np.array([x*depthAtPixel, y*depthAtPixel, depthAtPixel,1])

        
        ## Nk
        ## M
        for u in range( dHeight -1): #Neighbouring
            for v in range( dWidth -1 ): #Neighbouring
                if 0 < self.Vk[u, v, 2] < 255:
                    n = np.cross( (self.Vk[u+1, v, :] - self.Vk[u,v,:]), self.Vk[u, v+1,:] - self.Vk[u,v,:])
                    if(np.linalg.norm(n)!=0 and Dk[u,v]!=0):
                        self.Nk[u, v, :] = n/np.linalg.norm(n)
                        self.M[u,v] = 1


        ## Supposed to be HxW,3 , H,W,3 , H,W
