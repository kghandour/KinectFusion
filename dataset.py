import os
import numpy as np

class Dataset():
    def __init__(self, dataset_name):
        root_folder = os.path.dirname(os.path.abspath(os.getcwd()))
        kinect_dataset_path = os.path.join(root_folder, "Data", dataset_name)
        self.make_dataset(kinect_dataset_path)
        self.processNextFrame()

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

    def processNextFrame(self):
        depth_file = open(self.depth_path, "r")
        rgb_file = open(self.rgb_path, "r")
        current_frame = 0
        groundtruth_file = open(self.groundtruth_path, "r")

        rgb_timestamps = []
        rgb_images = []
        depth_timestamps = []
        depth_images = []


        for depth_frame,color_frame in zip(depth_file, rgb_file):
            if(current_frame<3):
                current_frame +=1
                continue
            print("Processing frame ["+str(current_frame-2)+"/"+str(self.number_frames-3)+"]")
            depth_split = depth_frame.split()
            color_split = color_frame.split()

            if(len(depth_split)<2):
                raise Exception("File does not follow typical format")
            if(len(color_split)<2):
                raise Exception("File does not follow typical format")
            depth_timestamps.append(depth_split[0])
            depth_images.append(depth_split[1])
            rgb_timestamps.append(color_split[0])
            rgb_images.append(color_split[1])
            current_frame +=1