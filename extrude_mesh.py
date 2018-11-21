import argparse
import h5py
import numpy as np
from double_pipe import numpy_to_dolfin, sort_mesh, double_mesh
import dolfin as df
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Extrude a 2D mesh")
    parser.add_argument("mesh_file", type=str, help="Mesh filename")
    parser.add_argument("dz", type=float, help="Thickness")
    parser.add_argument("-Nz", type=int, default=0, help="Doublings")
    args = parser.parse_args()
    return args


def read_mesh(mesh_file):
    with h5py.File(mesh_file, "r") as h5f:
        node = np.array(h5f["mesh/coordinates"])
        elem = np.array(h5f["mesh/topology"])
    return node, elem


def basename(mesh_file):
    name_prefix = mesh_file.split(".h5")[0] + "_extruded"
    return name_prefix


def write_hdf(mesh, mesh_file):
    h5f = df.HDF5File(mesh.mpi_comm(), basename(mesh_file) + ".h5", "w")
    h5f.write(mesh, "mesh")
    h5f.close()


def write_xdmf(mesh, mesh_file):
    xdmff = df.XDMFFile(mesh.mpi_comm(), basename(mesh_file) + "_show.xdmf")
    xdmff.write(mesh)
    xdmff.close()


def main():
    args = parse_args()

    node2d, face_1 = read_mesh(args.mesh_file)

    node_1 = np.zeros((len(node2d), 3))
    node_2 = np.zeros_like(node_1)
    node_1[:, :2] = node2d[:, :]
    node_2[:, :2] = node2d[:, :]
    node_2[:, 2] = args.dz
    node = np.vstack((node_1, node_2))

    face_2 = np.copy(face_1)
    face_2[:, :] += len(node2d)
    prisms = np.hstack((face_1, face_2))

    elem = np.zeros((3*len(prisms), 4)).astype(int)
    for i, p in enumerate(prisms):
        pp = np.vstack((p[0:4], p[1:5], p[2:6]))
        elem[3*i:3*(i+1), :] = pp

    node, elem = sort_mesh(node, elem)

    for i in range(args.Nz):
        node, elem = double_mesh(node, elem)
        node[:, 2] /= 2.

    mesh = numpy_to_dolfin(node, elem)

    write_hdf(mesh, args.mesh_file)
    write_xdmf(mesh, args.mesh_file)


if __name__ == "__main__":
    main()
