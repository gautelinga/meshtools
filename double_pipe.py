from __future__ import print_function
import argparse
import h5py
import os
import numpy as np
import progressbar as pb
from mpi4py import MPI
import dolfin as df


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def parse_args():
    parser = argparse.ArgumentParser(description="Double a pipe mesh.")
    parser.add_argument("mesh_file", type=str, help="Mesh filename")
    parser.add_argument("--reflect", action="store_true",
                        help="Reflect mesh while doubling.")
    parser.add_argument("--axis", type=str, default="z",
                        help="Axis to double along")
    return parser.parse_args()


def add_nodes(node_dict, node):
    i_unique = len(node_dict)
    for x in node:
        xtup = tuple(x.tolist())
        if xtup not in node_dict:
            node_dict[xtup] = i_unique
            i_unique += 1


def remove_safe(path):
    """ Remove file in a safe way. """
    if rank == 0 and os.path.exists(path):
        os.remove(path)


def numpy_to_dolfin(nodes, elements):
    """ Convert nodes and elements to a dolfin mesh object. """
    tmpfile = "tmp.h5"
    if rank == 0:
        with h5py.File(tmpfile, "w") as h5f:
            cell_indices = h5f.create_dataset(
                "mesh/cell_indices", data=np.arange(len(elements)),
                dtype='int64')
            topology = h5f.create_dataset(
                "mesh/topology", data=elements, dtype='int64')
            coordinates = h5f.create_dataset(
                "mesh/coordinates", data=nodes, dtype='float64')
            topology.attrs["celltype"] = np.string_("tetrahedron")
            topology.attrs["partition"] = np.array([0], dtype='uint64')

    comm.Barrier()

    mesh = df.Mesh()
    h5f = df.HDF5File(mesh.mpi_comm(), tmpfile, "r")
    h5f.read(mesh, "mesh", False)
    h5f.close()

    comm.Barrier()
    remove_safe(tmpfile)
    return mesh
            

def sort_mesh(node, elem):
    sortids = np.argsort(node[:, 2])
    sortids_inv = np.zeros_like(sortids)
    for i, j in enumerate(sortids):
        sortids_inv[j] = i

    node_sorted = node[sortids, :]
    elem_sorted = sortids_inv[elem.flatten()].reshape(elem.shape)

    node = node_sorted
    elem = elem_sorted
    return node, elem


def main():
    args = parse_args()

    if not os.path.exists(args.mesh_file):
        exit("Couldn't find file")

    with h5py.File(args.mesh_file, "r") as h5f:
        node = np.array(h5f["mesh/coordinates"])
        elem = np.array(h5f["mesh/topology"])

    print("Node:", node.shape)
    print("Elem:", elem.shape)

    if args.axis == "x":
        node_map = [1, 2, 0]
    elif args.axis == "y":
        node_map = [2, 0, 1]
    else:  # args.axis == "z":
        node_map = [0, 1, 2]
    node_map = zip(range(len(node_map)), node_map)

    node_cp = np.zeros_like(node)
    for i, j in node_map:
        node_cp[:, i] = node[:, j]
    node[:, :] = node_cp[:, :]

    node, elem = sort_mesh(node, elem)

    x_max = node.max(0)
    x_min = node.min(0)

    if bool(np.sum(node[:, 2] == x_max[2]) !=
            np.sum(node[:, 2] > x_max[2]-1e-7)):
        print(x_max[2])
        node[node[:, 2] > x_max[2]-1e-7, 2] = x_max[2]

    node_new = np.zeros_like(node)
    elem_new = np.zeros_like(elem)
    node_new[:, :] = node[:, :]
    elem_new[:, :] = elem[:, :]
    if not args.reflect:
        node_new[:, 2] += x_max[2]-x_min[2]
    else:
        node_new[:, 2] = 2*x_max[2]-node_new[:, 2]
        node_new, elem_new = sort_mesh(node_new, elem_new)

    glue_ids_old = np.argwhere(node[:, 2] == x_max[2]).flatten()
    glue_ids_new = np.argwhere(node_new[:, 2] == x_max[2]).flatten()

    print(len(glue_ids_old))
    print(len(glue_ids_new))

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

    print("Node_out:", node_out.shape)
    print("Elem_out:", elem_out.shape)

    mesh = numpy_to_dolfin(node_out, elem_out)

    name_prefix = args.mesh_file.split(".h5")[0] + "_double"

    h5f = df.HDF5File(
        mesh.mpi_comm(),
        name_prefix + ".h5", "w")
    h5f.write(mesh, "mesh")
    h5f.close()

    xdmff = df.XDMFFile(mesh.mpi_comm(),
                        name_prefix + "_show.xdmf")
    xdmff.write(mesh)
    # xdmff.close()


if __name__ == "__main__":
    main()
