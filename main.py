from dataset import Dataset
from virtualSensor import VirtualSensor
import numpy as np
import math
import os
from parser import Parser
from plotlytest import PlotViewer



if __name__ == "__main__":
    # Assumes the same file structure as the exercises. Meaning before the Exercises/KinectFusion/main.py
    # Dataset folder will be in Exercises/Data/
    dataset = Dataset("rgbd_dataset_freiburg1_xyz") 
    sensor = VirtualSensor(dataset, 800)
    parser = Parser(sensor)
    parser.process()
    plot_view = PlotViewer()


    

