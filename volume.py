class Volume():
    def __init__(self, tsdf_volume, weights_volume, rgb_volume):
        self.tsdf_volume = tsdf_volume
        self.weights_volume = weights_volume
        self.rgb_volume = rgb_volume