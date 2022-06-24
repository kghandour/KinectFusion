from dataset import Dataset
from virtualSensor import VirtualSensor

if __name__ == "__main__":
    # Assumes the same file structure as the exercises. Meaning before the Exercises/KinectFusion/main.py
    # Dataset folder will be in Exercises/Data/
    dataset = Dataset("rgbd_dataset_freiburg1_xyz") 
    sensor = VirtualSensor(dataset)

    
    sensor.processNextFrame()
