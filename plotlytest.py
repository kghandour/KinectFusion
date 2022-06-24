import plotly.graph_objects as go
import numpy as np


class PlotViewer():
    def __init__(self):
        pts = np.loadtxt(np.DataSource().open("mesh.txt"))
        #pts = np.loadtxt(np.DataSource().open("https://raw.githubusercontent.com/plotly/datasets/master/mesh_dataset.txt"))
        x, y, z = pts.T

        fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z,
                        alphahull=5,
                        opacity=0.4,
                        color='cyan')])
        fig.show()