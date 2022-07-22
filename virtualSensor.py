from dataset import Dataset
from PIL import Image
import numpy as np
import os
import math
from camera_sensors import CamDetails

class VirtualSensor():
    def __init__(self, dataset, increment):
        self.rgb_timestamps = dataset.rgb_timestamps
        self.rgb_images_path = dataset.rgb_images_path
        self.depth_timestamps = dataset.depth_timestamps
        self.depth_images_path = dataset.depth_images_path
        self.trajectory_timestamps = dataset.trajectory_timestamps
        self.trajectory = dataset.trajectory
        self.kinect_dataset_path = dataset.kinect_dataset_path
        self.currentIdx = -1
        self.increment = increment   ## Change increment to skip frames in the middle

        self.m_colorImageWidth = CamDetails.colorWidth
        self.m_colorImageHeight = CamDetails.colorHeight
        self.m_depthImageWidth = CamDetails.depthWidth
        self.m_depthImageHeight = CamDetails.depthHeight
        
        self.m_colorIntrinsics = CamDetails.colorIntrinsics


        self.m_depthIntrinsics = CamDetails.depthIntrinsics

        self.m_colorExtrinsics = np.identity(4)
        self.m_depthExtrinsics = np.identity(4)

        self.m_depthFrame = np.full((self.m_depthImageWidth, self.m_depthImageHeight), 0.5)
        self.m_colorFrame = np.full((self.m_colorImageWidth, self.m_colorImageHeight),255)

    def processNextFrame(self):
        if(self.currentIdx==-1):
            self.currentIdx =0
        else:
            self.currentIdx += self.increment

        if(self.currentIdx>=len(self.rgb_images_path)):
            return False
        
        print("ProcessNextFrame "+str(self.currentIdx)+" | " + str(len((self.rgb_images_path))))

        self.rgbImage = np.asarray(Image.open(os.path.join(self.kinect_dataset_path,self.rgb_images_path[self.currentIdx])))
        self.dImageRaw = Image.open(os.path.join(self.kinect_dataset_path,self.depth_images_path[self.currentIdx]))
        self.dImage = np.asarray(self.dImageRaw)
        self.dImage = np.where(self.dImage==0, -math.inf, self.dImage * 1 / 5000)

        
        ## Finds the nearest neighbouring trajectory
        timestamp = self.depth_timestamps[self.currentIdx]
        min_val = math.inf
        idx = 0
        for i in range(len(self.trajectory)):
            d = abs(float(self.trajectory_timestamps[i]) - float(timestamp))
            if (min_val > d) :
                min_val = d
                idx = i

        self.currentTrajectory = self.trajectory[idx]

        return True


