import dolfin as df
import mshr as mshr
import numpy as np
from mpi4py import MPI
import argparse
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def barbelltube1(res):
    rightbell = mshr.Cylinder(df.Point(0.0, 0.0, -25.),
                              df.Point(0.0, 0.0, -35.),
                              10., 10., 400)
    tube = mshr.Cylinder(df.Point(0., 0., 0.),
                         df.Point(0., 0., -25.),
                         5, 5, 400)
    domain = rightbell + tube

    mesh = mshr.generate_mesh(domain, res)
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int,
                        default=50, help='The resolution ')
    parser.add_argument('--pl', action="store_true",
                        help='plot the mesh')
    args = parser.parse_args(sys.argv[1:])

    mesh = barbelltube1(args.res)

    if args.pl:
        df.plot(mesh)
        df.interactive()

    filename = "halfbarbell_res{}.h5".format(args.res)

    hdf5 = df.HDF5File(mesh.mpi_comm(), filename, "w")
    hdf5.write(mesh, "mesh")
    hdf5.close()


if __name__ == "__main__":
    main()
