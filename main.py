from turtle import down
from dataset import Dataset
from virtualSensor import VirtualSensor
import numpy as np
import math
import os

def WriteMesh(vertices_position, vertices_color, width, height, filename):
    f = open(filename, "w")
    edgeThreshold = 0.01

    nVertices = width*height
    nFaces = 0

    for i in range(height):
        for j in range(width):
            currentVertex = vertices_position[i,j]
            if(currentVertex[0]!=-math.inf):
                if(j+1 < width and i+1 < height):
                    rightVertex = vertices_position[i,j+1]
                    downVertex = vertices_position[i+1,j]
                    dist1 = np.linalg.norm(currentVertex - rightVertex)
                    dist2 = np.linalg.norm(currentVertex - downVertex)
                    dist3 = np.linalg.norm(downVertex - rightVertex)
                    if(dist1 < edgeThreshold and dist2 < edgeThreshold and dist3 < edgeThreshold and (dist1>0 and dist2 >0 and dist3>0)):
                        nFaces += 1
                if(j-1 > 0 and i+1 < height):
                    downVertex = vertices_position[i+1,j]
                    downLeftVertex = vertices_position[i+1, j-1]
                    dist1 = np.linalg.norm(currentVertex - downLeftVertex)
                    dist2 = np.linalg.norm(currentVertex - downVertex)
                    dist3 = np.linalg.norm(downVertex - downLeftVertex)
                    if(dist1 < edgeThreshold and dist2 < edgeThreshold and dist3 < edgeThreshold and (dist1>0 and dist2 >0 and dist3>0)):
                        nFaces += 1
                    
    f.write("COFF\n")
    f.write("# numVertices numFaces numEdges\n")
    f.write(str(nVertices) + " " + str(nFaces) + " 0\n")
    f.write("# list of Vertices\n")
    f.write("# X Y Z R G B A\n")

    for i in range(height):
        for j in range(width):
            if(vertices_position[i][j][0]!=-math.inf and vertices_position[i][j][1]!=-math.inf and vertices_position[i][j][2]!=-math.inf):
                f.write(str(vertices_position[i][j][0]) +" "+
                 str(vertices_position[i][j][1])+" "+
                 str(vertices_position[i][j][2])+" "+
                 str(vertices_color[i][j][0])+" "+
                 str(vertices_color[i][j][1])+" "+
                 str(vertices_color[i][j][2])+" "+
                 str(vertices_color[i][j][3])+"\n")
            else:
                f.write(str(0) +" "+
                 str(0)+" "+
                 str(0)+" "+
                 str(vertices_color[i][j][0])+" "+
                 str(vertices_color[i][j][1])+" "+
                 str(vertices_color[i][j][2])+" "+
                 str(vertices_color[i][j][3])+"\n")

    f.write("# list of Faces\n")
    f.write("# nVerticesPerFace idx0 idx1 idx2 ...\n")
    for i in range(height):
        for j in range(width):
            currentVertex = vertices_position[i,j]
            if(currentVertex[0]!=-math.inf):
                if(j+1 < width and i+1 < height):
                    rightVertex = vertices_position[i,j+1]
                    downVertex = vertices_position[i+1,j]
                    dist1 = np.linalg.norm(currentVertex - rightVertex)
                    dist2 = np.linalg.norm(currentVertex - downVertex)
                    dist3 = np.linalg.norm(downVertex - rightVertex)
                    if(dist1 < edgeThreshold and dist2 < edgeThreshold and dist3 < edgeThreshold and (dist1>0 or dist2 >0 or dist3>0)):
                        f.write("3 "+str(j+ i*width)+" "+ str(j+1+ i*width)+" "+ str(j+(i+1)*width)+"\n")
                if(j-1 > 0 and i+1 < height):
                    downVertex = vertices_position[i+1,j]
                    downLeftVertex = vertices_position[i+1, j-1]
                    dist1 = np.linalg.norm(currentVertex - downLeftVertex)
                    dist2 = np.linalg.norm(currentVertex - downVertex)
                    dist3 = np.linalg.norm(downVertex - downLeftVertex)
                    if(dist1 < edgeThreshold and dist2 < edgeThreshold and dist3 < edgeThreshold and (dist1>0 or dist2 >0 or dist3>0)):
                        f.write("3 "+str(j+ i*width)+" "+ str(j+(i+1)*width)+" "+ str(j-1+(i+1)*width)+"\n")
    
    f.close()

if __name__ == "__main__":
    # Assumes the same file structure as the exercises. Meaning before the Exercises/KinectFusion/main.py
    # Dataset folder will be in Exercises/Data/
    fileBaseOut = "mesh_"
    dataset = Dataset("rgbd_dataset_freiburg1_xyz") 
    sensor = VirtualSensor(dataset, 10)


    while(sensor.processNextFrame()):
        depthMap = sensor.dImage
        colorMap = sensor.rgbImage

        depthIntrinsics = sensor.m_depthIntrinsics
        depthIntrinsicsInv = np.linalg.inv(depthIntrinsics)

        fX = depthIntrinsics[0, 0]
        fY = depthIntrinsics[1, 1]
        cX = depthIntrinsics[0, 2]
        cY = depthIntrinsics[1, 2]

        depthExtrinsicsInv = np.linalg.inv(sensor.m_depthExtrinsics)

        trajectory = sensor.currentTrajectory
        trajectoryInv = np.linalg.inv(trajectory)
        
        cameraSpace = np.zeros((sensor.m_depthImageHeight, sensor.m_depthImageWidth, 4))

        for i in range(sensor.m_depthImageHeight):
            for j in range(sensor.m_depthImageWidth):
                x = (j - cX) / fX
                y = (i - cY) / fY
                depthAtPixel = depthMap[i,j]
                if(depthAtPixel != -math.inf):
                    cameraSpace[i,j] = np.array([x*depthAtPixel, y*depthAtPixel, depthAtPixel, 1])
                else:
                    cameraSpace[i,j] = np.array([-math.inf, -math.inf, -math.inf, -math.inf])


        vertices_position = np.zeros((sensor.m_depthImageHeight, sensor.m_depthImageWidth, 4))
        vertices_color = np.full((sensor.m_depthImageHeight, sensor.m_depthImageWidth, 4), 255)
        for i in range(sensor.m_depthImageHeight):
            for j in range(sensor.m_depthImageWidth):
                depthAtPixel = depthMap[i,j]
                if(depthAtPixel != -math.inf):
                    vertices_position[i,j] = trajectoryInv.dot(cameraSpace[i,j])
                    vertices_color[i,j] = [colorMap[i][j][0],colorMap[i][j][1],colorMap[i][j][2],255]
                else:
                    vertices_position[i,j] = np.array([-math.inf, -math.inf, -math.inf, -math.inf])
                    vertices_color[i,j] = np.array([0,0,0,0])

        
        fileName = os.path.join("mesh_out/",str(fileBaseOut)+str(sensor.currentIdx)+".off")
        WriteMesh(vertices_position, vertices_color, sensor.m_colorImageWidth, sensor.m_colorImageHeight, fileName)

        

