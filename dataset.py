import os
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import torch
from config import config



class Dataset():
    def __init__(self, dataset_name):
        root_folder = os.path.dirname(os.path.abspath(os.getcwd()))
        self.kinect_dataset_path = os.path.join(root_folder, "Data", dataset_name)
        self.make_dataset(self.kinect_dataset_path)
        self.preprocess(self.kinect_dataset_path)

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
        self.rgb_number_frames = sum(1 for line in rgb_file)
        if(self.number_frames != self.rgb_number_frames):
            print(self.number_frames, self.rgb_number_frames)
            raise Exception("Depth and Color files frames do not match. Please make sure that they have the same number of lines")

        depth_file.close()
        rgb_file.close()
        
        

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


        for depth_frame,color_frame in zip(depth_file, rgb_file):
            if(current_frame<3):
                current_frame +=1
                continue
            depth_split = depth_frame.split()
            color_split = color_frame.split()
            
            if(len(depth_split)<2):
                raise Exception("File does not follow typical format")
            if(len(color_split)<2):
                raise Exception("File does not follow typical format")
            

            ## Loading depth images
            self.depth_timestamps.append(depth_split[0])
            self.depth_images_path.append(depth_split[1])

            ## Loading RGB Images
            self.rgb_timestamps.append(color_split[0])
            self.rgb_images_path.append(color_split[1])

            
        ## Loading trajectory details
        current_frame = 0
        for trajectory_frame in groundtruth_file:
            if(current_frame<3):
                current_frame +=1
                continue
            trajectory_split = trajectory_frame.split()
            if(len(trajectory_split)<2):
                raise Exception("File does not follow typical format")
            
            self.trajectory_timestamps.append(trajectory_split[0])
            temp_traj = trajectory_split[1:]
            trajMatrix = torch.eye(4)
            trajMatrix[0,3] = float(temp_traj[0])
            trajMatrix[1,3] = float(temp_traj[1])
            trajMatrix[2,3] = float(temp_traj[2])
            rotMatrix = R.from_quat(temp_traj[3:]).as_matrix()
            trajMatrix[0:3,0:3] = torch.from_numpy(rotMatrix)
            if(np.linalg.norm(rotMatrix)==0):
                break

            trajInverse = torch.linalg.inv(trajMatrix)
            self.trajectory.append(trajInverse)