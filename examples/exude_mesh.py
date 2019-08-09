import argparse
import dolfin as df
from meshtools.io import numpy_to_dolfin, dolfin_file_to_numpy
from meshtools.volume import exude_2d_mesh_to_3d


def parse_args():
    parser = argparse.ArgumentParser(description="Exude a 2D mesh")
    parser.add_argument("mesh_file", type=str, help="Mesh filename")
    parser.add_argument("dz", type=float, help="Thickness")
    parser.add_argument("-Nz", type=int, default=0, help="Doublings")
    args = parser.parse_args()
    return args


def basename(mesh_file):
    name_prefix = mesh_file.split(".h5")[0] + "_exuded"
    return name_prefix


def write_hdf(mesh, mesh_file):
    with df.HDF5File(mesh.mpi_comm(),
                     basename(mesh_file) + ".h5", "w") as h5f:
        h5f.write(mesh, "mesh")


def write_xdmf(mesh, mesh_file):
    with df.XDMFFile(mesh.mpi_comm(),
                     basename(mesh_file) + "_show.xdmf") as xdmff:
        xdmff.write(mesh)


def main():
    args = parse_args()

    node2d, elem2d = dolfin_file_to_numpy(args.mesh_file)

    node, elem = exude_2d_mesh_to_3d(node2d, elem2d, args.dz, args.Nz)

    mesh = numpy_to_dolfin(node, elem)

    write_hdf(mesh, args.mesh_file)
    write_xdmf(mesh, args.mesh_file)


if __name__ == "__main__":
    main()
