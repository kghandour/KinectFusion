from dataset import Dataset
from PIL import Image
import numpy as np
import os
import math

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

        self.m_colorImageWidth = 640
        self.m_colorImageHeight = 480
        self.m_depthImageWidth = 640
        self.m_depthImageHeight = 480
        
        self.m_colorIntrinsics = np.array([525.0, 0.0, 319.5, 0.0, 525.0, 239.5,0.0, 0.0, 1]).reshape((3,3))


        self.m_depthIntrinsics = self.m_colorIntrinsics

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


