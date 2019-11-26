import numpy as np
from scipy import ndimage
from itertools import product
from .volume import condense_mesh
from .io import voxel_mesh_to_dolfin
import skimage


def grid(bounding_box, N):
    x = [np.linspace(bounding_box[dim, 0], bounding_box[dim, 1], N[dim])
         for dim in range(3)]
    X, Y, Z = np.meshgrid(*x, indexing="ij")
    return X, Y, Z


def trim_voxels(a):
    """ Remove dead space around data. """
    bw = a > 0
    a_01 = bw.sum(axis=(0, 1))
    a_02 = bw.sum(axis=(0, 2))
    a_12 = bw.sum(axis=(1, 2))

    ind_z = np.where(a_01)[0]
    ind_y = np.where(a_02)[0]
    ind_x = np.where(a_12)[0]

    A = a[ind_x[0]:ind_x[-1]+1,
          ind_y[0]:ind_y[-1]+1,
          ind_z[0]:ind_z[-1]+1]
    return A


def laplacian_filter(S, k, periodic=False):
    S_cp = np.zeros([i+2 for i in S.shape])
    S_cp[1:-1, 1:-1, 1:-1] = S
    if periodic:
        S_cp[0, :, :] = S_cp[-2, :, :]
        S_cp[:, 0, :] = S_cp[:, -2, :]
        S_cp[:, :, 0] = S_cp[:, :, -2]
        S_cp[-1, :, :] = S_cp[1, :, :]
        S_cp[:, -1, :] = S_cp[:, 1, :]
        S_cp[:, :, -1] = S_cp[:, :, 1]
    else:
        S_cp[0, :, :] = S_cp[1, :, :]
        S_cp[:, 0, :] = S_cp[:, 1, :]
        S_cp[:, :, 0] = S_cp[:, :, 1]
        S_cp[-1, :, :] = S_cp[-2, :, :]
        S_cp[:, -1, :] = S_cp[:, -2, :]
        S_cp[:, :, -1] = S_cp[:, :, -2]
    S_cp2 = np.zeros_like(S)
    S_cp2[:, :, :] = (1-6*k)*S[:, :, :]
    S_cp2[:, :, :] += k*(S_cp[:-2, 1:-1, 1:-1]
                         + S_cp[2:, 1:-1, 1:-1]
                         + S_cp[1:-1, :-2, 1:-1]
                         + S_cp[1:-1, 2:, 1:-1]
                         + S_cp[1:-1, 1:-1, :-2]
                         + S_cp[1:-1, 1:-1, 2:])
    return S_cp2


def extract_voxel_faces(cells):
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


def get_subcluster(iic, D, d=0):
    li, lj, lk = iic.shape
    if np.size(D) == 1:
        Di = Dj = Dk = D
    else:  # no.size(D) == 3
        Di, Dj, Dk = D
    if np.size(d) == 1:
        di = dj = dk = d
    else:
        di, dj, dk = d

    return iic[li//2-Di+di:li//2+Di+1+di,
               lj//2-Dj+dj:lj//2+Dj+1+dj,
               lk//2-Dk+dk:lk//2+Dk+1+dk]


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


def get_connected_clusters(clusters, cluster_conn):
    iic = np.zeros_like(clusters[0], dtype=bool)
    for i, cluster in enumerate(clusters):
        if cluster_conn[i]:
            iic[cluster] = True
    return iic


def extract_backbone(bw, axis=0):
    clusters, cluster_conn = get_clusters(bw, axis)
    return get_connected_clusters(clusters, cluster_conn)


def _generate_nodes(N, dim):
    X = np.meshgrid(*(range(N+1),)*dim)
    nodes = list(zip(*tuple([list(X[i].flatten()) for i in range(dim)])))
    return nodes


def _compute_node_dict(nodes):
    coord_to_id = dict()
    for i, node in enumerate(nodes):
        coord_to_id[node] = i
    return coord_to_id


def _build_cells(cell_coords, coord_to_id, dim):
    unit_cell = np.array([list(reversed(el)) for el in
                          product([0, 1], repeat=dim)], dtype=float)
    cells = np.zeros((len(cell_coords), 2**dim), dtype=int)
    for i, cell_coord in enumerate(cell_coords):
        X_loc = np.array(cell_coord)*np.ones((2**dim, 1)) + unit_cell
        cells[i, :] = np.array([coord_to_id[tuple(x)] for x in X_loc],
                               dtype=int)
    return cells


def voxels_to_voxel_mesh(iic):
    dim = len(iic.shape)
    N = np.max(iic.shape)
    nodes = _generate_nodes(N, dim)
    coord_to_id = _compute_node_dict(nodes)

    X_cell = np.meshgrid(*(range(N),)*dim, indexing="ij")

    li1 = [X_cell[d][iic] for d in range(dim)]
    li = list(zip(*tuple(li1)))
    cell_coords = np.array(li)
    cells = _build_cells(cell_coords, coord_to_id, dim)
    nodes = np.array(nodes)/N
    nodes, cells = condense_mesh(nodes, cells)
    return nodes, cells, cell_coords


def voxels_to_dolfin(iic):
    """Generate voxel mesh (hexahedral or quadrilateral) in Dolfin format
    from voxel data."""
    nodes, cells, cell_coords = voxels_to_voxel_mesh(iic)
    mesh = voxel_mesh_to_dolfin(nodes, cells)
    return mesh, nodes, cells, cell_coords


def tif2vox(filename):
    a = skimage.io.imread(filename)
    A = trim_voxels(a)
    return A


def refine_voxels(S, m=2):
    I, J, K = S.shape
    S_2 = np.zeros((m*I, m*J, m*K), dtype=bool)
    for i in range(m):
        for j in range(m):
            for k in range(m):
                S_2[i::m, j::m, k::m] = S
    return S_2
