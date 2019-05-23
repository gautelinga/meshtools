import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mplcolors
from common import numpy_to_dolfin
import dolfin as df
import operator


def connected(s, ax):
    dims = list(range(len(s.shape)))
    dims.remove(ax)

    x = np.sum(s, axis=tuple(dims)) > 0
    return x[0] & x[-1]


def get_clusters(bw, axis=0):
    labeled, num_objects = ndimage.label(bw)
    clusters = [labeled == i for i in range(1, num_objects)]
    cluster_conn = [connected(cluster, axis) for cluster in clusters]

    return clusters, cluster_conn


def only_connected(clusters, cluster_conn):
    iic = np.zeros_like(clusters[0], dtype=bool)
    for i, cluster in enumerate(clusters):
        if cluster_conn[i]:
            iic[cluster] = True
    return iic


def generate_cluster(N, dim, p, axis=0):
    size = (N,)*dim
    if N == 1:
        return np.ones(size, dtype=bool)
    
    conn = False
    while not conn:
        R = np.random.rand(*size)
        bw = R < p

        clusters, cluster_conn = get_clusters(bw, axis=axis)
        conn = np.any(cluster_conn)

    iic = only_connected(clusters, cluster_conn)
    return iic


def extract_backbone(bw, axis=0):
    clusters, cluster_conn = get_clusters(bw, axis)
    return only_connected(clusters, cluster_conn)


def plot_vox(cells, nodes, cell_coords, iic):
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


def plot_mesh(cells, nodes):
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

        faces, internal_faces, external_faces, face_to_cell = extract_faces(cells)
        boundary_faces = faces[external_faces]
        boundary_cells = np.array([face_to_cell[i][0] for i in external_faces])
        for face, cell in zip(boundary_faces, boundary_cells):
            face = face[[0, 1, 3, 2]]
            tri = Poly3DCollection([nodes[face]])
            tri.set_color(mplcolors.rgb2hex(np.ones(3)*cell/max(1, np.max(boundary_cells))))
            tri.set_edgecolor('k')
            ax.add_collection3d(tri)


def generate_nodes(N, dim):
    X = np.meshgrid(*(range(N+1),)*dim)
    nodes = list(zip(*tuple([list(X[i].flatten()) for i in range(dim)])))
    return nodes


def compute_node_dict(nodes):
    coord_to_id = dict()
    for i, node in enumerate(nodes):
        coord_to_id[node] = i
    return coord_to_id


def build_cells(cell_coords, coord_to_id, dim):
    unit_cell = np.array([list(reversed(el)) for el in
                          product([0, 1], repeat=dim)], dtype=float)
    cells = np.zeros((len(cell_coords), 2**dim), dtype=int)
    for i, cell_coord in enumerate(cell_coords):
        X_loc = np.array(cell_coord)*np.ones((2**dim, 1)) + unit_cell
        cells[i, :] = np.array([coord_to_id[tuple(x)] for x in X_loc],
                               dtype=int)
    return cells


def compute_mesh(iic):
    dim = len(iic.shape)
    N = np.max(iic.shape)
    nodes = generate_nodes(N, dim)
    coord_to_id = compute_node_dict(nodes)

    X_cell = np.meshgrid(*(range(N),)*dim)
    cell_coords = np.array(list(zip(*tuple([X_cell[i][iic]
                                            for i in range(dim)]))))
    cells = build_cells(cell_coords, coord_to_id, dim)
    nodes = np.array(nodes)/N
    return cells, nodes, cell_coords


def extract_faces(cells):
    face_dict = dict()
    nodes_loc = [(0, 1, 2, 3),
                 (0, 1, 4, 5),
                 (2, 3, 6, 7),
                 (4, 5, 6, 7),
                 (1, 3, 5, 7),
                 (0, 2, 4, 6)]
    for ic, cell in enumerate(cells):
        for ids in nodes_loc:
            face_loc = tuple(sorted(cell[list(ids)]))
            if face_loc in face_dict:
                face_dict[face_loc].append(ic)
            else:
                face_dict[face_loc] = [ic]

    faces = []
    face_to_cell = []
    internal_faces = []
    external_faces = []
    undisclosed_faces = []
    for j, (f_nodes, f_cells) in enumerate(face_dict.items()):
        faces.append(f_nodes)
        face_to_cell.append(f_cells)
        if len(f_cells) == 1:
            external_faces.append(j)
        elif len(f_cells) == 2:
            internal_faces.append(j)
        else:
            undisclosed_faces.append((j, f_cells))

    faces = np.array(faces)
    return faces, internal_faces, external_faces, face_to_cell


def prepare_indices(ids_before, ids_after):
    zipped = list(zip(ids_before, ids_after))
    zipped = sorted(zipped, key=operator.itemgetter(0))
    ids_before, ids_after = zip(*zipped)
    ids_before = np.array(ids_before)
    ids_after = np.array(ids_after)
    return ids_before, ids_after


def condense_mesh(cells, nodes):
    ids_before = np.unique(cells.flatten())
    ids_after = np.array(list(range(len(ids_before))))
    index = np.digitize(cells.ravel(), ids_before, right=True)
    cells_out = np.sort(ids_after[index].reshape(cells.shape), axis=1)
    nodes_out = nodes[ids_before, :]
    return cells_out, nodes_out


def convert_to_dolfin_mesh(cells, nodes):
    dim = nodes.shape[1]
    cells_2 = np.zeros_like(cells)
    cells_2[:, :] = cells[:, :]
    for i in range(dim-1):
        cells_2[:, 4*i+2] = cells[:, 4*i+3]
        cells_2[:, 4*i+3] = cells[:, 4*i+2]
    mesh = numpy_to_dolfin(nodes, cells_2, delete_tmp=True)
    return mesh


def generate(iic):
    cells, nodes, cell_coords = compute_mesh(iic)
    cells, nodes = condense_mesh(cells, nodes)
    mesh = convert_to_dolfin_mesh(cells, nodes)
    return mesh, cells, nodes, cell_coords


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make a periodic voxel structure.")
    parser.add_argument("-N", type=int, default=16, help="Number of voxels")
    parser.add_argument("-D", "--dim", type=int, default=2, help="Dimensions")
    parser.add_argument("-A", "--axis", type=int, default=0, help="Axis")
    parser.add_argument("--plot", action="store_true", help="Plot")
    parser.add_argument("--xdmf", action="store_true", help="Store XDMF")
    parser.add_argument("-o", "--outfile", type=str, default="", help="Outfile")
    parser.add_argument("-p", type=float, default=0.5, help="Percolation probability")
    args = parser.parse_args()

    iic = generate_cluster(args.N, args.dim, args.p, axis=args.axis)

    mesh, cells, nodes, cell_coords = generate(iic)

    if args.outfile is not "":
        fname = args.outfile.split(".")
        ext = fname[-1]
        if ext in ["h5", "hdf", "hdf5"]:
            h5f = df.HDF5File(mesh.mpi_comm(), args.outfile, "w")
            h5f.write(mesh, "mesh")
            h5f.close()

    if args.xdmf:
        xdmff = df.XDMFFile(mesh.mpi_comm(), "mesh.xdmf")
        xdmff.write(mesh)
        xdmff.close()

    if args.plot:
        plot_vox(cells, nodes*args.N, cell_coords, iic)
        plot_mesh(cells, nodes)
        plt.show()
