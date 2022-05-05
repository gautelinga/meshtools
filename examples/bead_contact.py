import argparse
import os
import dolfin as df
from meshtools.io import numpy_to_dolfin, dolfin_to_numpy
from meshtools.volume import double_mesh
import mshr
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Bead contact mesh.")
    parser.add_argument("mesh_file", type=str, help="Mesh filename")
    parser.add_argument("-Lx", type=float, default=1.5, help="Length x")
    parser.add_argument("-Ly", type=float, default=2.0, help="Length y")
    parser.add_argument("-H", type=float, default=1.0, help="Height")
    parser.add_argument("-R", type=float, default=0.5, help="Radius")
    parser.add_argument("-res", type=int, default=24, help="Resolution")
    parser.add_argument("-segments", type=int, default=100, help="Segments")
    parser.add_argument("-reps", type=float, default=0.1, help="Radius of regularizer")
    return parser.parse_args()


def main():
    args = parse_args()

    name_prefix = "{}".format(args.mesh_file.split(".h5")[0])

    Lx = args.Lx
    Ly = args.Ly
    H = args.H
    R = args.R

    pt_000 = df.Point(  0., Ly/2., H/2)
    pt_111 = df.Point(Lx/2,    0.,  0.)
    pt_001 = df.Point(  0., Ly/2.,  0.)

    cube = mshr.Box(pt_000, pt_111)
    sphere = mshr.Sphere(pt_000, R, segments=args.segments)
    geom = cube - sphere
    if args.reps > 0.:
        cyl = mshr.Cylinder(pt_000, pt_001, args.reps, args.reps, segments=args.segments)
        geom = geom - cyl

    mesh_part = mshr.generate_mesh(geom, args.res)

    with df.XDMFFile(mesh_part.mpi_comm(),
                     "{}_part_show.xdmf".format(name_prefix)) as xdmff:
        xdmff.write(mesh_part)

    node, elem = dolfin_to_numpy(mesh_part)

    print("Node:", node.shape)
    print("Elem:", elem.shape)

    for dim in [2]:
        node, elem = double_mesh(node, elem, dim, True)

    print("Node_out:", node.shape)
    print("Elem_out:", elem.shape)

    print("min:", node.min(0))
    print("max:", node.max(0))
    vol_enc = np.prod(node.max(0)-node.min(0))
    print("vol_enc =", vol_enc)

    mesh = numpy_to_dolfin(node, elem)

    vol = df.assemble(df.Constant(1.)*df.dx(domain=mesh))
    print("vol =", vol)
    print("porosity =", vol/vol_enc)

    with df.HDF5File(mesh.mpi_comm(), args.mesh_file, "w") as h5f:
        h5f.write(mesh, "mesh")

    with df.XDMFFile(mesh.mpi_comm(),
                     "{}_show.xdmf".format(name_prefix)) as xdmff:
        xdmff.write(mesh)


if __name__ == "__main__":
    main()
