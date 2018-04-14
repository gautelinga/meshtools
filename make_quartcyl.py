import dolfin as df
import mshr
import matplotlib.pyplot as plt
import sys
import argparse


def quartcyl(Lx, Ly, Lz, R, res):
    A = df.Point(0., 0., 0.)
    B = df.Point(0., 0., Lz)
    C = df.Point(-Lx/2, -Ly/2, 0.)

    box = mshr.Box(C, B)
    cyl = mshr.Cylinder(A, B, R, R, 400)

    domain = box - cyl

    mesh = mshr.generate_mesh(domain, res)
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-Lx", type=float, default=30.,
                        help="Length along x")
    parser.add_argument("-Ly", type=float, default=30.,
                        help="Length along y")
    parser.add_argument("-Lz", type=float, default=12.,
                        help="Length along z")
    parser.add_argument("-R", type=float, default=4.,
                        help="Radius")
    parser.add_argument('--res', type=int,
                        default=25, help='The resolution ')
    parser.add_argument('--pl', action="store_true",
                        help='plot the mesh')
    args = parser.parse_args(sys.argv[1:])

    mesh = quartcyl(args.Lx, args.Ly, args.Lz, args.R, args.res)

    if args.pl:
        df.plot(mesh)
        plt.show()

    filename = "quartcyl_res{}.h5".format(args.res)

    hdf5 = df.HDF5File(mesh.mpi_comm(), filename, "w")
    hdf5.write(mesh, "mesh")
    hdf5.close()


if __name__ == "__main__":
    main()
