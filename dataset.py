import os
import numpy as np
from PIL import Image
import math

class Dataset():
    def __init__(self, dataset_name):
        root_folder = os.path.dirname(os.path.abspath(os.getcwd()))
        self.kinect_dataset_path = os.path.join(root_folder, "Data", dataset_name)
        self.make_dataset(self.kinect_dataset_path)
        self.preprocess(self.kinect_dataset_path)
        self.processNextFrame(self.kinect_dataset_path)

    def make_dataset(self, directory):
        self.depth_path = os.path.join(directory,"depth.txt")
        self.rgb_path = os.path.join(directory,"rgb.txt")
        self.groundtruth_path = os.path.join(directory,"groundtruth.txt")
        if(not os.path.exists(self.depth_path)):
            raise Exception("Depth.txt not found. Please make sure that you downloaded the dataset and is storing it in ../Data/")
        if(not os.path.exists(self.rgb_path)):
            raise Exception("RGB.txt not found. Please make sure that you downloaded the dataset and is storing it in ../Data/")
        if(not os.path.exists(self.groundtruth_path)):
            raise Exception("Groundtruth.txt not found. Please make sure that you downloaded the dataset and is storing it in ../Data/")

        depth_file = open(self.depth_path, "r")
        rgb_file = open(self.rgb_path, "r")
        groundtruth_file = open(self.groundtruth_path, "r")

        self.number_frames = sum(1 for line in depth_file)
        if(self.number_frames != sum(1 for line in rgb_file)):
            raise Exception("Depth and Color files frames do not match. Please make sure that they have the same number of lines")

        depth_file.close()
        rgb_file.close()
        
        self.m_colorImageWidth = 640
        self.m_colorImageHeight = 480
        self.m_depthImageWidth = 640
        self.m_depthImageHeight = 480
        
        self.m_colorIntrinsics = np.array([525.0, 0.0, 319.5, 0.0, 525.0, 239.5,0.0, 0.0, 1]).reshape((3,3))


        self.m_depthIntrinsics = self.m_colorIntrinsics

        self.m_colorExtrinsics = np.identity(3)

        self.m_depthFrame = np.full((self.m_depthImageWidth, self.m_depthImageHeight), 0.5)
        self.m_colorFrame = np.full((self.m_colorImageWidth, self.m_colorImageHeight),255)

    def preprocess(self, directory):
        depth_file = open(self.depth_path, "r")
        rgb_file = open(self.rgb_path, "r")
        groundtruth_file = open(self.groundtruth_path, "r")

        self.rgb_timestamps = []
        self.rgb_images_path = []
        self.depth_timestamps = []
        self.depth_images_path = []
        self.trajectory_timestamps = []
        self.trajectory = []
        current_frame = 0
        for depth_frame,color_frame,trajectory_frame in zip(depth_file, rgb_file,groundtruth_file):
            if(current_frame<3):
                current_frame +=1
                continue
            # print("Processing frame ["+str(current_frame-2)+"/"+str(self.number_frames-3)+"]")
            depth_split = depth_frame.split()
            color_split = color_frame.split()
            trajectory_split = trajectory_frame.split()

            if(len(depth_split)<2):
                raise Exception("File does not follow typical format")
            if(len(color_split)<2):
                raise Exception("File does not follow typical format")
            if(len(trajectory_split)<2):
                raise Exception("File does not follow typical format")

            ## Loading depth images
            self.depth_timestamps.append(depth_split[0])
            self.depth_images_path.append(depth_split[1])

            ## Loading RGB Images
            self.rgb_timestamps.append(color_split[0])
            self.rgb_images_path.append(color_split[1])

            ## Loading trajectory details
            self.trajectory_timestamps.append(trajectory_split[0])
            self.trajectory.append(trajectory_split[1:])

    def processNextFrame(self, directory):
        pass
            # color_images.append(np.asarray(Image.open(os.path.join(directory,color_split[1]))))

            # dImage = np.asarray(Image.open(os.path.join(directory,depth_split[1])))
            # dImage = np.where(dImage==0, -math.inf, dImage * 1 / 5000)
            # depth_images.append(dImage)

            # timestamp = depth_timestamps[current_frame]
            # min = math.inf
            # idx = 0
            # for (i = 0; i < m_trajectory.size(); ++i) {
            #     double d = abs(m_trajectoryTimeStamps[i] - timestamp);
            #     if (min > d) {
            #         min = d;
            #         idx = i;
            #     }
            # }
            # m_currentTrajectory = m_trajectory[idx];

            # current_frame +=1

            ##End processing frames

