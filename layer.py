import cv2
from cv2 import destroyWindow
import numpy as np
from PIL import ImageOps, Image
from sympy import true
from camera_sensors import CamDetails

from transforms import Transforms
from torchvision import transforms
import torch
import open3d as o3d
import time
from config import config
from numba import jit
import math

class Layer():
    def __init__(self,  depthImage, rgbImage, sensor):
        self.sensor = sensor
        self.depthImage = depthImage
        self.rgbImage = np.swapaxes(rgbImage, 0,1)
        self.rgbImageRaw = self.rgbImage
        self.dHeight = self.depthImage.shape[1]
        self.dWidth = self.depthImage.shape[0]

        self.Dk = []

        self.compNormalsVerticesAndMask()


    def compNormalsVerticesAndMask(self):
        ### Surface Measuremenet

        ## Dk  --------
        ##TODO: Make sure if Depth must be divided by 5000 or not. I normalized the depth directly to 255 instead of dividing by 5000
        depth_arr = np.array(self.depthImage)
        # normalize = np.array(self.depthImage)/np.max(self.depthImage)*255
        # image = Image.fromarray(normalize)
        # imgConv = ImageOps.grayscale(image)
        # Dk = cv2.bilateralFilter(np.asarray(imgConv), 15,20,20)
        transform_toTensor = transforms.ToTensor()

        self.Dk = transform_toTensor(depth_arr).permute(1,2,0)
        # Vk = Camera space

        self.Vk = Transforms.screen2cam(self.Dk, vis_3d=False)
        # Nk
        # M
        self.Nk, self.M = Layer.normals(self.dWidth, self.dHeight, np.array(self.Vk, dtype=np.float32))        ## Supposed to be HxW,3 , H,W,3 , H,W
        self.Vk = self.Vk[self.M == 1].reshape(-1,3)
        self.Nk = np.nan_to_num(self.Nk[self.M == 1].reshape(-1,3), nan=0)
        self.rgbImage = self.rgbImage[self.M == 1].reshape(-1,3)



    @staticmethod
    @jit(nopython=True)
    def normals(w:int, h:int, Vk):
        Nk = np.zeros(( w,  h, 3), dtype=np.float32)
        M = np.zeros((w, h))
        for u in range( w-1 ): #Neighbouring
            for v in range( h-1  ): #Neighbouring
                n = np.zeros((3, ), dtype=np.float32)
                if Vk[u, v, 2] != -math.inf :
                    n = np.cross((Vk[u+1, v,:] - Vk[u,v,:]), Vk[u, v+1,:] - Vk[u,v,:])
                if(np.linalg.norm(n)!=0):
                    Nk[u, v, :] = n/np.linalg.norm(n)
                    M[u,v] = 1
        return Nk, M
