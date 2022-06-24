from dataset import Dataset

class VirtualSensor():
    def __init__(self, dataset):
        self.rgb_timestamps = dataset.rgb_timestamps
        self.rgb_images_path = dataset.rgb_images_path
        self.depth_timestamps = dataset.depth_timestamps
        self.depth_images_path = dataset.depth_images_path
        self.trajectory_timestamps = dataset.trajectory_timestamps
        self.trajectory = dataset.trajectory
        self.kinect_dataset_path = dataset.kinect_dataset_path

    def processNextFrame(self):
        print("Processing now ")
        print(self.kinect_dataset_path)