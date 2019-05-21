import numpy as np
import argparse
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mplcolors
import meshio
import dolfin as df


def connected(s, ax):
    dims = list(range(len(s.shape)))
    dims.remove(ax)
    
    x = np.sum(s, axis=tuple(dims)) > 0
    return x[0] & x[-1]


def generate_cluster(N, dim):
    size = (N,)*dim

    conn = False
    while not conn:
        R = np.random.rand(*size)
        bw = R < 0.6

        labeled, num_objects = ndimage.label(bw)
        clusters = [labeled == i for i in range(1, num_objects)]
        cluster_conn = [connected(cluster, args.axis) for cluster in clusters]

        conn = np.any(cluster_conn)

    iic = np.zeros_like(bw, dtype=bool)
    for i, cluster in enumerate(clusters):
        if cluster_conn[i]:
            iic[cluster] = True
    return iic


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
            tri.set_color(mplcolors.rgb2hex(np.ones(3)*cell/np.max(boundary_cells)))
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


def reshuffle(a):
    dim = a.shape[1]
    b = np.zeros_like(a)
    b[:, :] = a[:, :]
    # for i in range(dim-1):
    #     b[2*i, :] = a[2*i+1, :]
    #     b[2*i+1, :] = a[2*i, :]
    return b


def build_cells(cell_coords, coord_to_id, dim):
    unit_cell = reshuffle(np.array([list(reversed(el)) for el in
                                    product([0, 1], repeat=dim)], dtype=float))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make a periodic voxel structure.")
    parser.add_argument("-N", type=int, default=16, help="Number of voxels")
    parser.add_argument("-D", "--dim", type=int, default=2, help="Dimensions")
    parser.add_argument("-A", "--axis", type=int, default=0, help="Axis")
    args = parser.parse_args()

    mesh_type = ["vertex", "line", "quad", "hexahedron"]
    
    iic = generate_cluster(args.N, args.dim)

    cells, nodes, cell_coords = compute_mesh(iic)

    dim = args.dim
    cells_2 = np.zeros_like(cells)
    cells_2[:, :] = cells[:, :]
    for i in range(dim-1):
        cells_2[:, 4*i+2] = cells[:, 4*i+3]
        cells_2[:, 4*i+3] = cells[:, 4*i+2]
    cells_out = {
        mesh_type[args.dim]: cells_2
    }
    mesh = meshio.Mesh(nodes, cells_out)
    meshio.write("foo.xml", mesh)

    m = df.Mesh("foo.xml")
    xdmff = df.XDMFFile(m.comm_world(), "mesh.xdmf")
    xdmff.write(m, "mesh")
    xdmff.close()
    
    plot_vox(cells, nodes*args.N, cell_coords, iic)
    plot_mesh(cells, nodes)
    plt.show()
