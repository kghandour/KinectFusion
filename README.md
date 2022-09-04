# KinectFusion
## Setup Environment

```
conda create --name kinect python=3.10
conda activate kinect
pip install -r requirements.txt
```

## Pre-requirements
1- Download the dataset freiburg1_xyz and store it in Data folder in the parent folder of the KinectFusion/ folder. 
https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz 

2- Create folder mesh_out in the base of KinectFusion

3- Your directory structure should be as follows
```
Project/ 
-> Data/rgbd_dataset_freiburg1_xyz/
-> KinectFusion/
    -> mesh_out/
    -> code files
```

4- To run simply 
```
python main.py
```
## Documentation
### main.py
Takes in the arguments and initializes the dataset loader, VirtualSensor and Parser
possible arguments are
```
--visualizeTSDF (Visualizes the TSDF after 700 frame)
--ground (Uses the ground truth instead of the ICP)
```

### dataset.py
Loads our dataset by parsing the folder content and storing the paths for the images as well as storing the ground truth

### virtualSensor.py
Processes the frame by loading the depth and rgb images as well as loads the current trajectory ground truth

### camera_sensors.py
Stores the camera intrinsics and the image sizes

### kinect_parser.py
Initializes the TSDF Volume with our camera intrinsics, responsible for the main loop of the application by calling the virtualsensor to process the next frame, then calls the Layer to generate the vertices and normal maps, then initializes the TSDF fusion first frame, and following frames it calls the ICP then integrates the TSDF. It is also responsible for exporting and viewing the TSDF

### layer.py
Responsible for the surface measurements.

### transforms.py 
mainstreams the transformation between spaces

### tsdf.py
Responsible for the TSDF implementation and integration

### icp.py
responsible for the pose estimate using ICP whether frame to frame or frame to volume.

