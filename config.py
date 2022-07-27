import torch


_torchDevice = None
_visualize = False
_visualizeTSDF = False

def getTorchDevice():
    return _torchDevice

def getVisualizeBool():
    return _visualize

def getVisualizeTSDFBool():
    return _visualizeTSDF

def setTorchDevice(device):
    _torchDevice = device

def setVisualize(vis):
    _visualize = vis

def setVisualizeTSDF(vis_tsdf):
    _visualizeTSDF = vis_tsdf