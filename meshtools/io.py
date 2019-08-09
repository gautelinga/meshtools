import os
from .cmd import mpi_is_root, mpi_barrier
import numpy as np
import dolfin as df
import h5py


def remove_safe(path):
    """ Remove file in a safe way. """
    if mpi_is_root() and os.path.exists(path):
        os.remove(path)


def generate_tetgen_input(filename, node, face):
    with open("{}.node".format(filename), 'w') as outfile:
        # Information header
        outfile.write("#\n"
                      "# TetGen input file\n"
                      "#\n"
                      "# rough.node\n"
                      "#\n"
                      "# Rough duct or channel in .smesh format.\n"
                      "#\n"
                      "# Created by Gaute Linga\n"
                      "#\n\n")

        num_nodes = np.size(node, 0)
        np.savetxt(outfile, np.array([(num_nodes, 3, 0, 0)]), fmt='%d')

        node_out = np.zeros((num_nodes, 4))
        node_out[:, 0] = np.arange(1, num_nodes+1)
        node_out[:, 1:4] = node
        np.savetxt(outfile,
                   node_out,
                   fmt='%d\t%1.10f\t%1.10f\t%1.10f')
    with open("{}.smesh".format(filename), "w") as outfile:
        # Write header
        outfile.write("#\n"
                      "# TetGen input file\n"
                      "#\n"
                      "# rough.smesh\n"
                      "#\n"
                      "# Rough duct or channel in .smesh format.\n"
                      "#\n"
                      "# Created by Gaute Linga\n"
                      "#\n\n"
                      "# part 1, node list\n"
                      "#   '0' indicates that the node "
                      "list is stored in '{}.node'\n".format(filename))
        np.savetxt(outfile, np.array([(0, 3, 0, 0)]), fmt='%d')
        outfile.write("\n"
                      "# part 2, facet list\n")

        num_faces = np.size(face, 0)
        np.savetxt(outfile, np.array([(num_faces, 0)]), fmt='%d')
        outfile.write("\n")

        face_out = np.zeros((num_faces, 4))
        face_out[:, 0] = 3
        face_out[:, 1:4] = face + 1
        np.savetxt(outfile, face_out, fmt='%d', delimiter='\t')
        outfile.write("\n"
                      "# part 3, hole list\n"
                      "0\n"
                      "\n"
                      "# part 4, region list\n"
                      "0")


def save_mesh(file_in, file_out, rotate=False, rotate_xz=False,
              xdmf=True, hdf=False):
    mesh = df.Mesh(file_in)
    if rotate:
        x = mesh.coordinates()
        xtemp = np.zeros_like(x)
        xtemp[:, :] = x[:, :]
        x[:, 1] = xtemp[:, 2]
        x[:, 2] = xtemp[:, 1]

    if rotate_xz:
        x = mesh.coordinates()[:]
        x = x[:, [1, 2, 0]]
        mesh.coordinates()[:] = x

    if xdmf:
        xdmf_file = df.XDMFFile(mesh.mpi_comm(), file_out + "_show.xdmf")
        xdmf_file.write(mesh)
        xdmf_file.close()

    if hdf:
        h5_file = df.HDF5File(mesh.mpi_comm(), file_out + ".h5", "w")
        h5_file.write(mesh, "/mesh")
        h5_file.close()


def numpy_to_dolfin(nodes, elements, delete_tmp=True):
    """ Convert nodes and elements to a dolfin mesh object. """
    tmpfile = "tmp.h5"

    dim = nodes.shape[1]
    npts = elements.shape[1]
    if dim == 0:
        celltype_str = "point"
    elif dim == 1:
        celltype_str = "interval"
    elif dim == 2:
        if npts == 3:
            celltype_str = "triangle"
        elif npts == 4:
            celltype_str = "quadrilateral"
    elif dim == 3:
        if npts == 4:
            celltype_str = "tetrahedron"
        elif npts == 8:
            celltype_str = "hexahedron"

    if mpi_is_root():
        with h5py.File(tmpfile, "w") as h5f:
            cell_indices = h5f.create_dataset(
                "mesh/cell_indices", data=np.arange(len(elements)),
                dtype='int64')
            topology = h5f.create_dataset(
                "mesh/topology", data=elements, dtype='int64')
            coordinates = h5f.create_dataset(
                "mesh/coordinates", data=nodes, dtype='float64')
            topology.attrs["celltype"] = np.string_(celltype_str)
            topology.attrs["partition"] = np.array([0], dtype='uint64')

    mpi_barrier()

    mesh = df.Mesh()
    h5f = df.HDF5File(mesh.mpi_comm(), tmpfile, "r")
    h5f.read(mesh, "mesh", False)
    h5f.close()

    mpi_barrier()
    if delete_tmp:
        remove_safe(tmpfile)
    return mesh


def dolfin_file_to_numpy(mesh_file):
    with h5py.File(mesh_file, "r") as h5f:
        nodes = np.array(h5f["mesh/coordinates"])
        elements = np.array(h5f["mesh/topology"])
    return nodes, elements


def dolfin_to_numpy(mesh):
    tmpfile = "tmp.h5"
    with df.HDF5File(mesh.mpi_comm(), tmpfile, "w") as h5f:
        h5f.write(mesh, "mesh")
    nodes, elements = dolfin_file_to_numpy(tmpfile)
    remove_safe(tmpfile)
    return nodes, elements


def voxel_mesh_to_dolfin(nodes, cells):
    dim = nodes.shape[1]
    cells_2 = np.zeros_like(cells)
    cells_2[:, :] = cells[:, :]
    for i in range(dim-1):
        cells_2[:, 4*i+2] = cells[:, 4*i+3]
        cells_2[:, 4*i+3] = cells[:, 4*i+2]
    mesh = numpy_to_dolfin(nodes, cells_2, delete_tmp=True)
    return mesh
