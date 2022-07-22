import layer as Layer
from transforms import Transforms
import tsdf as TSDF
import virtualSensor as Sensor
import numpy as np

class Raycast():
    def __init__(self, sensor: Sensor, tsdf: TSDF, layer: Layer):
        self.sensor = sensor
        self.tsdf = tsdf
        self.layer = layer

    def cast(self):
        # Vk = Camera space
        Vk = self.layer.Vk
        # Vk_homo = np.hstack((Vk, np.ones(Vk.shape[1])))
        # print(Vk_homo.shape)

        trajectory =  self.sensor.currentTrajectory
        worldSpace = Transforms.cam2world(Vk, trajectory)
        print(worldSpace.shape)