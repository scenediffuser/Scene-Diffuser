import numpy as np
from plotly import graph_objects as go

colors = [
    'blue', 'red', 'yellow', 'pink', 'gray', 'orange'
]

def plot_mesh(mesh, color='lightblue', opacity=1.0):
    return go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color=color, opacity=opacity)

def plot_hand(verts, faces, color='lightpink', opacity=1.0):
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color, opacity=opacity)

def plot_contact_points(pts, grad, color='lightpink'):
    pts = pts.detach().cpu().numpy()
    grad = grad.detach().cpu().numpy()
    grad = grad / np.linalg.norm(grad, axis=-1, keepdims=True)
    return go.Cone(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], u=-grad[:, 0], v=-grad[:, 1], w=-grad[:, 2], anchor='tip',
                   colorscale=[(0, color), (1, color)], sizemode='absolute', sizeref=0.2, opacity=0.5)

def plot_point_cloud(pts, color='lightblue', mode='markers'):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode=mode,
        marker=dict(
            color=color,
            size=3.
        )
    )


occ_cmap = lambda levels, thres=0.: [f"rgb({int(255)},{int(255)},{int(255)})" if x > thres else
                           f"rgb({int(0)},{int(0)},{int(0)})" for x in levels.tolist()]


def plot_point_cloud_occ(pts, color_levels=None):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        marker={
            'color': occ_cmap(color_levels),
            'size': 3,
            'opacity': 1
        }
    )


contact_cmap = lambda levels, thres=0.: [f"rgb({int(255 * (1 - x))},{int(255 * (1 - x))},{int(255 * (1 - x))})" if x >= thres else
                                         f"rgb({int(0)},{int(0)},{int(0)})" for x in levels.tolist()]

def plot_point_cloud_cmap(pts, color_levels=None):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        marker={
            'color': contact_cmap(color_levels),
            'size': 3.5,
            'opacity': 1
        }
    )


normal_color_map = lambda levels, thres=0., color_scale=8.: [f"rgb({int(255 * (color_scale * x[0]))},{int(255 * (color_scale * x[1]))},{int(255 * (color_scale * x[2]))})" if x[0] >= thres else
                                                             f"rgb({int(0)},{int(0)},{int(0)})" for x in levels.tolist()]


def plot_normal_map(pts, normal):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        marker={
            'color': normal_color_map(np.abs(normal)),
            'size': 3.5,
            'opacity': 1
        }
    )