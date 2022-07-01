import os
import numpy as np
import math
import open3d as o3d

class Parser():
    def __init__(self,  sensor):
        self.sensor =  sensor
        self.fileBaseOut = "mesh_"

    def cleanUp(self, vertices_position, vertices_color, width, height):
        vert_pos = []
        vert_col = []
        for i in range(height):
            for j in range(width):
                if(vertices_position[i][j][0]!=-math.inf and vertices_position[i][j][1]!=-math.inf and vertices_position[i][j][2]!=-math.inf):
                    vert_pos.append(vertices_position[i][j][:3])
                    vert_col.append(vertices_color[i][j][:3])
        return np.array(vert_pos), np.array(vert_col)

    def visualize(self, vert_pos, vert_col):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vert_pos)
        pcd.colors = o3d.utility.Vector3dVector(vert_col/255)
        # o3d.io.write_point_cloud("./mesh_out/sync.ply", pcd)

        # Load saved point cloud and visualize it
        # pcd_load = o3d.io.read_point_cloud("./mesh_out/sync.ply")
        # print(pcd_load)
        o3d.visualization.draw_geometries([pcd])

    def WriteMesh(self, vertices_position, vertices_color, width, height, filename):
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

    def one_loop(self):
        depthMap =  self.sensor.dImage
        colorMap =  self.sensor.rgbImage

        depthIntrinsics =  self.sensor.m_depthIntrinsics
        depthIntrinsicsInv = np.linalg.inv(depthIntrinsics)

        fX = depthIntrinsics[0, 0]
        fY = depthIntrinsics[1, 1]
        cX = depthIntrinsics[0, 2]
        cY = depthIntrinsics[1, 2]

        depthExtrinsicsInv = np.linalg.inv( self.sensor.m_depthExtrinsics)

        trajectory =  self.sensor.currentTrajectory
        trajectoryInv = np.linalg.inv(trajectory)

        cameraSpace = np.zeros(( self.sensor.m_depthImageHeight,  self.sensor.m_depthImageWidth, 4))

        for i in range( self.sensor.m_depthImageHeight):
            for j in range( self.sensor.m_depthImageWidth):
                x = (j - cX) / fX
                y = (i - cY) / fY
                depthAtPixel = depthMap[i,j]
                if(depthAtPixel != -math.inf):
                    cameraSpace[i,j] = np.array([x*depthAtPixel, y*depthAtPixel, depthAtPixel, 1])
                else:
                    cameraSpace[i,j] = np.array([-math.inf, -math.inf, -math.inf, -math.inf])


        vertices_position = np.zeros(( self.sensor.m_depthImageHeight,  self.sensor.m_depthImageWidth, 4))
        vertices_color = np.full(( self.sensor.m_depthImageHeight,  self.sensor.m_depthImageWidth, 4), 255)
        for i in range( self.sensor.m_depthImageHeight):
            for j in range( self.sensor.m_depthImageWidth):
                depthAtPixel = depthMap[i,j]
                if(depthAtPixel != -math.inf):
                    vertices_position[i,j] = trajectoryInv.dot(cameraSpace[i,j])
                    vertices_color[i,j] = [colorMap[i][j][0],colorMap[i][j][1],colorMap[i][j][2],255]
                else:
                    vertices_position[i,j] = np.array([-math.inf, -math.inf, -math.inf, -math.inf])
                    vertices_color[i,j] = np.array([0,0,0,0])


        fileName = os.path.join("mesh_out/",str(self.fileBaseOut)+str( self.sensor.currentIdx)+".off")
        # self.WriteMesh(vertices_position, vertices_color,  self.sensor.m_colorImageWidth,  self.sensor.m_colorImageHeight, fileName)
        vert_pos, vert_col = self.cleanUp(vertices_position, vertices_color,  self.sensor.m_colorImageWidth,  self.sensor.m_colorImageHeight)
        return vert_pos, vert_col

    def process(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        i = 0
        pcd = o3d.geometry.PointCloud()
        while( self.sensor.processNextFrame()):
            vert_pos, vert_col = self.one_loop()
            pcd.points = o3d.utility.Vector3dVector(vert_pos)
            pcd.colors = o3d.utility.Vector3dVector(vert_col/255)
            if(i==0):
                vis.add_geometry(pcd)
            else:
                vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            i+=1
        vis.destroy_window()
            # self.visualize(vert_pos, vert_col)


        