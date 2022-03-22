import argparse
import os
import dolfin as df
from meshtools.io import numpy_to_dolfin, dolfin_to_numpy
from meshtools.volume import double_mesh
import mshr
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Double/flip a mesh.")
    parser.add_argument("mesh_file", type=str, help="Mesh filename")
    parser.add_argument("-L", type=float, default=1.0, help="Length")
    parser.add_argument("-R", type=float, default=0.4, help="Radius")
    parser.add_argument("-res", type=int, default=48, help="Resolution")
    parser.add_argument("-segments", type=int, default=100, help="Segments")
    parser.add_argument("-reps", type=float, default=0.0, help="Radius of regularizer")
    return parser.parse_args()


def main():
    args = parse_args()

    name_prefix = "{}".format(args.mesh_file.split(".h5")[0])

    L = args.L
    R = args.R

    pt1 = df.Point(0., 0., 0.)
    pt2 = df.Point(L/2, L/2, L/2)
    pt3 = df.Point(L/2, L/2, 0.)
    pt4 = df.Point(L/2, 0., L/2)
    pt5 = df.Point(0., L/2, L/2)
    pt6 = df.Point(L/4, L/4, 0.)
    pt7 = df.Point(L/4, 0., L/4)
    pt8 = df.Point(0., L/4, L/4)
    pt9 = df.Point(L/4, L/4, L/2)
    pt10 = df.Point(L/4, L/2, L/4)
    pt11 = df.Point(L/2, L/4, L/4)

    cube = mshr.Box(pt1, pt2)
    sphere1 = mshr.Sphere(pt1, R, segments=args.segments)
    sphere2 = mshr.Sphere(pt3, R, segments=args.segments)
    sphere3 = mshr.Sphere(pt4, R, segments=args.segments)
    sphere4 = mshr.Sphere(pt5, R, segments=args.segments)
    geom = cube - sphere1 - sphere2 -sphere3 - sphere4
    if args.reps > 0.:
        sphere5 = mshr.Sphere(pt6, args.reps, segments=args.segments)
        sphere6 = mshr.Sphere(pt7, args.reps, segments=args.segments)
        sphere7 = mshr.Sphere(pt8, args.reps, segments=args.segments)
        sphere8 = mshr.Sphere(pt9, args.reps, segments=args.segments)
        sphere9 = mshr.Sphere(pt10, args.reps, segments=args.segments)
        sphere10 = mshr.Sphere(pt11, args.reps, segments=args.segments)
        geom = geom - sphere5 - sphere6 - sphere7 - sphere8 -sphere9 - sphere10

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
