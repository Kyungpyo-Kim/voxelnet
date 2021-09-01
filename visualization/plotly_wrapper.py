import plotly.graph_objs as go
from utils.utils import *


def goLayout(title):
    return go.Layout(
        scene=dict(aspectmode='data'),
        title=title,
    )


def goScatterLidar(lidar, sampling=1):
    return go.Scatter3d(
        x=lidar['x'][::sampling],
        y=lidar['y'][::sampling],
        z=lidar['z'][::sampling],
        mode='markers',
        marker=dict(
            size=1,
            colorscale='Jet',
            opacity=0.8,
            color=lidar['i'],
            colorbar=dict(thickness=20, title='intensity'),
        )
    )


def goCube(box, color='#DC143C', opacity=0.6):
    x = box['x']
    y = box['y']
    z = box['z']
    h = box['h']
    w = box['w']
    l = box['l']
    yaw = box['yaw']
    import numpy as np
    x, y, z = getBBoxCorners3D(x, y, z, w, l, h, yaw)
    return go.Mesh3d(
        # 8 vertices of a cube
        x=x,
        y=y,
        z=z,
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        color=color,
        opacity=opacity,
        flatshading=True)


def goFigure(data, layout):
    fig = go.Figure(data=data, layout=layout)
    fig.show()
