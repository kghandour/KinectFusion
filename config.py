
class config():

    _torchDevice = None
    _visualize = False
    _visualizeTSDF = False
    _debug = False
    _groundTruth = False

    @staticmethod
    def getTorchDevice():
        return config._torchDevice

    @staticmethod
    def getVisualizeBool():
        return config._visualize

    @staticmethod
    def getVisualizeTSDFBool():
        return config._visualizeTSDF

    def getDebug():
        return config._debug

    def useGroundTruth():
        return config._groundTruth

    @staticmethod
    def setTorchDevice(device):
        config._torchDevice = device

    @staticmethod
    def setVisualize(vis):
        config._visualize = vis

    @staticmethod
    def setVisualizeTSDF(vis_tsdf):
        config._visualizeTSDF = vis_tsdf

    @staticmethod
    def setDebug(debug):
        config._debug = debug

    @staticmethod
    def setGroundTruth(groundTruth):
        config._groundTruth = groundTruth