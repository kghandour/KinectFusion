from dataset import Dataset
from virtualSensor import VirtualSensor
import numpy as np
import math
import os
from parser import Parser
import torch
import argparse

argparser= argparse.ArgumentParser(description='Parser Arguments for KinectFusion')
argparser.add_argument('--visualize', action='store_true', help='Visualizes using Open3D')
argparser.add_argument('--visualizeTSDF', action='store_true', help='Visualizes using Open3D')

args = argparser.parse_args()
visualize = args.visualize
visualizeTSDF = args.visualizeTSDF

device = torch.device("cpu")

def getVisualize():
    return visualize

def getVisualizeTSDF():
    return visualizeTSDF

def checkTorchDevice():
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def getTorchDevice():
    return device

if __name__ == "__main__":
    device = checkTorchDevice()
    # Assumes the same file structure as the exercises. Meaning before the Exercises/KinectFusion/main.py
    # Dataset folder will be in Exercises/Data/
    dataset = Dataset("rgbd_dataset_freiburg1_xyz") 
    sensor = VirtualSensor(dataset,1)
    parser = Parser(sensor)
    parser.process()
