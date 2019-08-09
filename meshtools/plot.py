from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import numpy as np
from .voxels import extract_voxel_faces


def plot_mesh(node, face):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_trisurf(node[:, 0], node[:, 1], node[:, 2],
                    triangles=face, cmap=plt.cm.viridis)

    X = node[:, 0]
    Y = node[:, 1]
    Z = node[:, 2]
    max_range = np.array([X.max()-X.min(),
                          Y.max()-Y.min(),
                          Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def plot_2d(node, face):
    plt.figure()
    plt.triplot(node[:, 0], node[:, 1], face)


def plot_vox(nodes, cells, cell_coords, iic):
    cell_ids = np.unique(cells)

    nodes_loc = nodes[cell_ids, :]-0.5

    dim = nodes.shape[1]
    if dim == 2:
        plt.imshow(iic)
        plt.plot(cell_coords[:, 0], cell_coords[:, 1], 'o')
        plt.plot(nodes_loc[:, 0], nodes_loc[:, 1], 'r.')
    elif dim == 3:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(iic, edgecolor='k')


def plot_voxel_mesh(nodes, cells):
    dim = nodes.shape[1]
    if dim == 2:
        patches = []
        for c in cells:
            p = Polygon(nodes[c[[0, 1, 3, 2]], :], True)
            patches.append(p)

        patch_coll = PatchCollection(patches, cmap=cm.viridis)
        colors = np.arange(len(patches), dtype=float)/len(patches)
        patch_coll.set_array(np.array(colors))

        fig, ax = plt.subplots()
        ax.add_collection(patch_coll)
    elif dim == 3:
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        (faces, internal_faces, external_faces,
         face_to_cell) = extract_voxel_faces(cells)
        boundary_faces = faces[external_faces]
        boundary_cells = np.array([face_to_cell[i][0] for i in external_faces])
        for face, cell in zip(boundary_faces, boundary_cells):
            face = face[[0, 1, 3, 2]]
            tri = Poly3DCollection([nodes[face]])
            tri.set_color(mplcolors.rgb2hex(
                np.ones(3)*cell/max(1, np.max(boundary_cells))))
            tri.set_edgecolor('k')
            ax.add_collection3d(tri)
