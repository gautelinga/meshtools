import argparse
import os
import dolfin as df
from meshtools.io import numpy_to_dolfin, dolfin_to_numpy
from meshtools.volume import double_mesh, shifted_mesh, stack_meshes
import mshr
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Baker's map mesh.")
    parser.add_argument("mesh_file", type=str, help="Mesh filename")
    parser.add_argument("-Lx", type=float, default=1.0, help="Length")
    parser.add_argument("-Ly", type=float, default=1.0, help="Radius")
    parser.add_argument("-Lz", type=float, default=1.0, help="Radius")
    parser.add_argument("-res", type=int, default=12, help="Resolution")
    return parser.parse_args()


def main():
    args = parse_args()

    name_prefix = "{}".format(args.mesh_file.split(".h5")[0])

    Lx = args.Lx
    Ly = args.Ly
    Lz = args.Lz

    pt1 = df.Point(0., 0., 0.)
    pt2 = df.Point(Lx/2, Ly/2, Lz/2)

    ccube = mshr.Box(pt1, pt2)

    mesh_part = mshr.generate_mesh(ccube, args.res)

    with df.XDMFFile(mesh_part.mpi_comm(),
                     "{}_part_show.xdmf".format(name_prefix)) as xdmff:
        xdmff.write(mesh_part)

    node, elem = dolfin_to_numpy(mesh_part)

    print("Node:", node.shape)
    print("Elem:", elem.shape)

    for dim in range(3):
        node, elem = double_mesh(node, elem, dim, True)

    #"""
    meshes = [(node, elem)]

    displacements = [
        [Lx, 0, 0], 
        [Lx, Ly, 0],
        [Lx, 2*Ly, 0],
        [Lx, -Ly, 0],
        [Lx, -2*Ly, 0],
        [Lx, 2*Ly, Lz],
        [Lx, 2*Ly, 2*Lz],
        [Lx, Ly, 2*Lz],
        [Lx, 0, 2*Lz],
        [2*Lx, 0, 2*Lz],
        [3*Lx, 0, 2*Lz],
        [3*Lx, 0, Lz],
        [3*Lx, 0, 0],
        [4*Lx, 0, 0],
        [3*Lx, 0, -Lz],
        [3*Lx, 0, -2*Lz],
        [2*Lx, 0, -2*Lz],
        [1*Lx, 0, -2*Lz],
        [1*Lx, -Ly, -2*Lz],
        [1*Lx, -2*Ly, -2*Lz],
        [1*Lx, -2*Ly, -Lz]
    ]
    for displacement in displacements:
        meshes.append(shifted_mesh(node, elem, displacement))

    node, elem = stack_meshes(meshes)
    #"""

    #node[:, 0] += Lx/2
    #node[:, 1] += Ly/2
    #node[:, 2] += Lz/2

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
