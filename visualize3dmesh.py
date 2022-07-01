# examples/Python/Basic/working_with_numpy.py

import copy
import numpy as np
import open3d as o3d

if __name__ == "__main__":

    file1 = open("D:\TUM\\"+str(3)+"DScanning\\mesh_0.off", 'r')

    line = file1.readline()
    line = file1.readline()
    verticesFaces = file1.readline().split(' ')
    line = file1.readline()
    line = file1.readline()
    vertices = verticesFaces[0]

    xyz = np.zeros(((int(vertices)), 3))
    xyzcolors = np.zeros(((int(vertices)), 3))
    

    
    for i in range(0, int(vertices)):
        line = file1.readline()
        line = line.split(' ')
        xyz[i][0] = line[0]
        xyz[i][1] = line[1]
        xyz[i][2] = line[2]
        xyzcolors[i][0] = line[3]
        xyzcolors[i][1] = line[4]
        xyzcolors[i][2] = line[5]

    file1.close()

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(xyzcolors/255)
    o3d.io.write_point_cloud("D:\TUM\\"+str(3)+"DScanning\\sync.ply", pcd)

    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud("D:\TUM\\"+str(3)+"DScanning\\sync.ply")
    o3d.visualization.draw_geometries([pcd_load])

   