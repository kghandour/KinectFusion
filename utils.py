import numpy as np
import math

def read_off(file):
    # if 'OFF' != file.readline().strip():
    #     raise('Not a valid OFF header')
    file.readline()
    file.readline()
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    file.readline()
    file.readline()
    verts = []
    colors = []
    for i_vert in range(n_verts):
        arr = []
        arr_color = []
        vertex_line = file.readline().strip().split(' ')
        for s in range(len(vertex_line)):
            if(s<3):
                arr.append(float(vertex_line[s]))
            else:
                arr_color.append(int(vertex_line[s]))
        verts.append(arr)
        colors.append(arr_color)
    file.readline()
    file.readline()
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces,colors


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

def cleanUp(self, vertices_position, vertices_color, width, height):
    vert_pos = []
    vert_col = []
    for i in range(height):
        for j in range(width):
            if(vertices_position[i][j][0]!=-math.inf and vertices_position[i][j][1]!=-math.inf and vertices_position[i][j][2]!=-math.inf):
                vert_pos.append(vertices_position[i][j][:3])
                vert_col.append(vertices_color[i][j][:3])
    return np.array(vert_pos), np.array(vert_col)