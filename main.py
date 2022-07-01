from dataset import Dataset
from virtualSensor import VirtualSensor
import numpy as np
import math
import os
from parser import Parser
# from plotlytest import PlotViewer
import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

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

def viewGL(verticies,surfaces, colors):
   glBegin(GL_QUADS)
   for surface in surfaces:
      for vertex in surface:
        glColor(colors[vertex][0],colors[vertex][1],colors[vertex][2])
        # glColor(255,0,0)
        glVertex3dv(verticies[vertex])
   glEnd()

# def square():
#     glBegin(GL_QUADS)
#     glVertex2f(100, 100)
#     glVertex2f(200, 100)
#     glVertex2f(200, 200)
#     glVertex2f(100, 200)
#     glEnd()

# def iterate():
#     glViewport(0, 0, 500, 500)
#     glMatrixMode(GL_PROJECTION)
#     glLoadIdentity()
#     glOrtho(0.0, 500, 0.0, 500, 0.0, 1.0)
#     glMatrixMode (GL_MODELVIEW)
#     glLoadIdentity()

def showScreen():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # Remove everything from screen (i.e. displays all white)
    # glLoadIdentity() # Reset all graphic/shape's position
    # iterate()
    # glColor3f(1.0, 0.0, 3.0)
    glScale(0.5, 0.5, 0.5)
    viewGL(verts, faces, colors)
    # glutSwapBuffers()
    glFlush()

if __name__ == "__main__":
    # Assumes the same file structure as the exercises. Meaning before the Exercises/KinectFusion/main.py
    # Dataset folder will be in Exercises/Data/
    dataset = Dataset("rgbd_dataset_freiburg1_xyz") 
    sensor = VirtualSensor(dataset, 40)
    parser = Parser(sensor)
    parser.process()
    # plot_view = PlotViewer()
    # verts, faces, colors = read_off(open(os.path.join("mesh_out/","mesh_0.off")))

    # glutInit() # Initialize a glut instance which will allow us to customize our window
    # glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA) # Set the display mode to be colored
    # glutInitWindowSize(1000, 1000)   # Set the width and height of your window
    # glutInitWindowPosition(0, 0)   # Set the position at which this windows should appear
    # wind = glutCreateWindow("Read OFF") # Give your window a title
    # glutDisplayFunc(showScreen)  # Tell OpenGL to call the showScreen method continuously
    # # glutIdleFunc(showScreen)     # Draw any graphics or shapes in the showScreen function at all times
    # glutMainLoop()  # Keeps the window created above displaying/running in a loop


    

