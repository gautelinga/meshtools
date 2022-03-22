import meshio
import pygalmesh
import numpy as np


def mesh_volume(verts, faces,
                edge_size=0.025,
                facet_angle=25.0,
                facet_size=0.025,
                facet_distance=0.025,
                cell_radius_edge_ratio=3.0,
                cell_size=1.0,
                feature_edges=None,
                detect_features=True,
                perturb=False,
                exude=False,
                lloyd=False,
                odt=False,
                verbose=True):
    cells = dict(triangle=np.array(faces))
    mesh = meshio.Mesh(verts, cells)

    if verbose:
        print("meshing volume")
    mesh = pygalmesh.generate_volume_mesh_from_surface_mesh(
        mesh,
        edge_size=edge_size,
        facet_angle=facet_angle,
        facet_size=facet_size,
        facet_distance=facet_distance,
        cell_radius_edge_ratio=cell_radius_edge_ratio,
        cell_size=cell_size,
        detect_features=detect_features,
        feature_edges=feature_edges,
        perturb=perturb,
        exude=exude,
        lloyd=lloyd,
        odt=odt,
        verbose=True)
    if verbose:
        print("done with that")
    verts = mesh.points
    elems = mesh.cells["tetra"]
    return verts, elems


def condense_mesh(nodes, cells):
    ids_before = np.unique(cells.flatten())
    ids_after = np.array(list(range(len(ids_before))))
    index = np.digitize(cells.ravel(), ids_before, right=True)
    cells_out = np.sort(ids_after[index].reshape(cells.shape), axis=1)
    nodes_out = nodes[ids_before, :]
    return nodes_out, cells_out


def _sort_mesh(node, elem):
    sortids = np.argsort(node[:, 2])
    sortids_inv = np.zeros_like(sortids)
    for i, j in enumerate(sortids):
        sortids_inv[j] = i

    node_sorted = node[sortids, :]
    elem_sorted = sortids_inv[elem.flatten()].reshape(elem.shape)

    node = node_sorted
    elem = elem_sorted
    return node, elem


def double_mesh(node, elem, axis="z", reflect=False, tol=1e-7):
    if axis in ("x", 0):
        node_map = [1, 2, 0]
    elif axis in ("y", 1):
        node_map = [2, 0, 1]
    else:  # axis in ("z", 2):
        node_map = [0, 1, 2]
    node_map = list(zip(range(len(node_map)), node_map))

    node_cp = np.zeros_like(node)
    for i, j in node_map:
        node_cp[:, i] = node[:, j]
    node[:, :] = node_cp[:, :]

    node, elem = _sort_mesh(node, elem)

    x_max = node.max(0)
    x_min = node.min(0)

    if bool(np.sum(node[:, 2] == x_max[2]) !=
            np.sum(node[:, 2] > x_max[2]-tol)):
        node[node[:, 2] > x_max[2]-tol, 2] = x_max[2]

    node_new = np.zeros_like(node)
    elem_new = np.zeros_like(elem)
    node_new[:, :] = node[:, :]
    elem_new[:, :] = elem[:, :]
    if not reflect:
        node_new[:, 2] += x_max[2]-x_min[2]
    else:
        node_new[:, 2] = 2*x_max[2]-node_new[:, 2]
        node_new, elem_new = _sort_mesh(node_new, elem_new)

    glue_ids_old = np.argwhere(node[:, 2] == x_max[2]).flatten()
    glue_ids_new = np.argwhere(node_new[:, 2] == x_max[2]).flatten()

    x_old = node[glue_ids_old, :]
    x_new = node_new[glue_ids_new, :]

    sortids_y_old = np.lexsort((x_old[:, 1], x_old[:, 0]))
    sortids_y_new = np.lexsort((x_new[:, 1], x_new[:, 0]))

    glue_ids_old_out = glue_ids_old[sortids_y_old]
    glue_ids_new_out = glue_ids_new[sortids_y_new]

    for i_old, i_new in zip(glue_ids_old_out, glue_ids_new_out):
        assert not any(node[i_old, :] - node_new[i_new, :])

    ids_map = np.zeros(len(node), dtype=int)
    for i_old, i_new in zip(glue_ids_old_out, glue_ids_new_out):
        ids_map[i_new] = i_old

    ids_map[len(glue_ids_old_out):] = np.arange(
        len(node), 2*len(node)-len(glue_ids_old_out))

    elem_new = ids_map[elem_new.flatten()].reshape(elem_new.shape)

    node_out = np.vstack((node, node_new[len(glue_ids_old_out):, :]))
    elem_out = np.vstack((elem, elem_new))

    node_cp = np.zeros_like(node_out)
    for i, j in node_map:
        node_cp[:, j] = node_out[:, i]
    node_out[:, :] = node_cp[:, :]
    return node_out, elem_out


def exude_2d_mesh_to_3d(node2d, elem2d, dz, Nz):
    node_1 = np.zeros((len(node2d), 3))
    node_2 = np.zeros_like(node_1)
    node_1[:, :2] = node2d[:, :]
    node_2[:, :2] = node2d[:, :]
    node_2[:, 2] = dz
    node = np.vstack((node_1, node_2))

    face_2 = np.copy(elem2d)
    face_2[:, :] += len(node2d)
    prisms = np.hstack((elem2d, face_2))

    elem = np.zeros((3*len(prisms), 4)).astype(int)
    for i, p in enumerate(prisms):
        pp = np.vstack((p[0:4], p[1:5], p[2:6]))
        elem[3*i:3*(i+1), :] = pp

    node, elem = _sort_mesh(node, elem)

    for i in range(Nz):
        node, elem = double_mesh(node, elem)
        node[:, 2] /= 2.

    return node, elem

def shifted_mesh(node, elem, displacement):
    node_out = np.copy(node)
    elem_out = np.copy(elem)
    for d in range(len(displacement)):
        node_out[:, d] += displacement[d]
    return node_out, elem_out

def stack_meshes(meshes):
    num_nodes = sum([node.shape[0] for node, elem in meshes])
    num_elems = sum([elem.shape[0] for node, elem in meshes])
    node_out = np.zeros((num_nodes, meshes[0][0].shape[1]))
    elem_out = np.zeros((num_elems, meshes[0][1].shape[1]), dtype=int)
    inode = 0
    ielem = 0
    for node, elem in meshes:
        inode_next = inode+node.shape[0]
        ielem_next = ielem+elem.shape[0]
        node_out[inode:inode_next, :] = node[:, :]
        elem_out[ielem:ielem_next, :] = elem[:, :] + inode
        inode = inode_next
        ielem = ielem_next

    K = 10000
    key2i = dict()
    old2new = np.zeros(node_out.shape[0], dtype=int)
    keep = np.zeros(node_out.shape[0], dtype=bool)
    i = 0
    for iv, v in enumerate(node_out):
        key = tuple([int(K*vi) for vi in v])
        if key in key2i:
            old2new[iv] = key2i[key]
        else:
            key2i[key] = i
            old2new[iv] = i
            keep[iv] = True
            i += 1
    node_out = node_out[keep, :]
    elem_out = old2new[elem_out]

    return node_out, elem_out