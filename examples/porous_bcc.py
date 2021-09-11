from __future__ import print_function
import argparse
import os
import dolfin as df
from meshtools.io import numpy_to_dolfin, dolfin_to_numpy
from meshtools.volume import double_mesh
import mshr


def parse_args():
    parser = argparse.ArgumentParser(description="Double/flip a mesh.")
    parser.add_argument("mesh_file", type=str, help="Mesh filename")
    parser.add_argument("-L", type=float, default=1.0, help="Length")
    parser.add_argument("-R", type=float, default=0.25, help="Radius")
    parser.add_argument("-res", type=int, default=48, help="Resolution")
    return parser.parse_args()


def main():
    args = parse_args()

    name_prefix = "{}".format(args.mesh_file.split(".h5")[0])

    L = args.L
    R = args.R

    pt1 = df.Point(0., 0., 0.)
    pt2 = df.Point(L/2, L/2, L/2)

    cube = mshr.Box(pt1, pt2)
    sphere1 = mshr.Sphere(pt1, R, segments=args.res)
    sphere2 = mshr.Sphere(pt2, R, segments=args.res)
    geom = cube - sphere1 - sphere2

    mesh_part = mshr.generate_mesh(geom, args.res)

    with df.XDMFFile(mesh_part.mpi_comm(),
                     "{}_part_show.xdmf".format(name_prefix)) as xdmff:
        xdmff.write(mesh_part)

    node, elem = dolfin_to_numpy(mesh_part)

    print("Node:", node.shape)
    print("Elem:", elem.shape)

    for dim in range(3):
        node, elem = double_mesh(node, elem, dim, True)

    print("Node_out:", node.shape)
    print("Elem_out:", elem.shape)

    mesh = numpy_to_dolfin(node, elem)

    with df.HDF5File(mesh.mpi_comm(), args.mesh_file, "w") as h5f:
        h5f.write(mesh, "mesh")

    with df.XDMFFile(mesh.mpi_comm(),
                     "{}_show.xdmf".format(name_prefix)) as xdmff:
        xdmff.write(mesh)


if __name__ == "__main__":
    main()
