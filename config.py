
class config():

    _torchDevice = None
    _visualize = False
    _visualizeTSDF = False
    
    @staticmethod
    def getTorchDevice():
        return config._torchDevice

    @staticmethod
    def getVisualizeBool():
        return config._visualize

    @staticmethod
    def getVisualizeTSDFBool():
        return config._visualizeTSDF

    @staticmethod
    def setTorchDevice(device):
        config._torchDevice = device

    @staticmethod
    def setVisualize(vis):
        config._visualize = vis

    @staticmethod
    def setVisualizeTSDF(vis_tsdf):
        config._visualizeTSDF = vis_tsdf
