import numpy as np

class CamDetails():
    colorWidth = 640
    colorHeight= 480
    depthWidth = 640
    depthHeight= 480
    colorIntrinsics = np.array([525.0, 0.0, 319.5, 0.0, 525.0, 239.5,0.0, 0.0, 1]).reshape((3,3))
    depthIntrinsics = colorIntrinsics
    fX = depthIntrinsics[0, 0]
    fY = depthIntrinsics[1, 1]
    cX = depthIntrinsics[0, 2]
    cY = depthIntrinsics[1, 2]