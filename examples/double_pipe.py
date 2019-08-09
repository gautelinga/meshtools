from __future__ import print_function
import argparse
import os
import dolfin as df
from meshtools.io import numpy_to_dolfin, dolfin_file_to_numpy
from meshtools.volume import double_mesh


def parse_args():
    parser = argparse.ArgumentParser(description="Double/flip a mesh.")
    parser.add_argument("mesh_file", type=str, help="Mesh filename")
    parser.add_argument("--reflect", action="store_true",
                        help="Reflect mesh while doubling.")
    parser.add_argument("--axis", type=str, default="z",
                        help="Axis to double along")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.mesh_file):
        exit("Couldn't find file")

    node, elem = dolfin_file_to_numpy(args.mesh_file)

    print("Node:", node.shape)
    print("Elem:", elem.shape)

    node_out, elem_out = double_mesh(node, elem, args.axis, args.reflect)

    print("Node_out:", node_out.shape)
    print("Elem_out:", elem_out.shape)

    mesh = numpy_to_dolfin(node_out, elem_out)

    name_prefix = "{}_double".format(args.mesh_file.split(".h5")[0])

    with df.HDF5File(mesh.mpi_comm(), "{}.h5".format(name_prefix),
                     "w") as h5f:
        h5f.write(mesh, "mesh")

    with df.XDMFFile(mesh.mpi_comm(),
                     "{}_show.xdmf".format(name_prefix)) as xdmff:
        xdmff.write(mesh)


if __name__ == "__main__":
    main()
