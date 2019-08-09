import dolfin as df
import numpy as np
import mshr
import argparse


def make_quarter_snoevsen_mesh(res, N, Lx, Ly, Lz, H, R, r):
    P1 = df.Point(-Lx/2, -Ly/2, 0.)
    P2 = df.Point(0., 0., Lz)
    Pmid = df.Point(0., 0., Lz/2)
    Pmidbelow = df.Point(0., 0., -H)
    Pmidjustbelow = df.Point(0., 0., -r)

    box = mshr.Box(P1, P2)
    box2 = mshr.Box(df.Point(0., -Ly/2, -H-R-r),
                    df.Point(Lx/2, Ly/2, Lz))
    box3 = mshr.Box(df.Point(-Lx/2, 0., -H-R-r),
                    df.Point(Lx/2, Ly/2, Lz))
    cyl = mshr.Cylinder(Pmid, Pmidbelow,
                        R, R, N)
    cyl2 = mshr.Cylinder(Pmid, Pmidjustbelow,
                         R+r, R+r, N)
    sph = mshr.Sphere(Pmidbelow, R+0.001, N)

    cyls = []
    for i in range(int(N/4+1)):
        theta1 = 2*np.pi*float(i)/(N) + np.pi
        theta2 = 2*np.pi*float(i+1)/(N) + np.pi
        cyls.append(
            mshr.Cylinder(df.Point((R+r)*np.sin(theta1),
                                   (R+r)*np.cos(theta1),
                                   -r),
                          df.Point((R+r)*np.sin(theta2),
                                   (R+r)*np.cos(theta2),
                                   -r),
                          r, r, N))

    domain = box + cyl + cyl2 + sph - box2 - box3
    for i in range(int(N/4+1)):
        domain -= cyls[i]

    mesh = mshr.generate_mesh(domain, res)

    return mesh


def main():
    parser = argparse.ArgumentParser(
        description="Make a quartermesh of 3D snoevsen.")
    parser.add_argument("-res", type=int, default=64, help="Resolution")
    parser.add_argument("-N", type=int, default=32, help="Number of segments")
    parser.add_argument("-Lx", type=float, default=6., help="Lx")
    parser.add_argument("-Ly", type=float, default=3., help="Ly")
    parser.add_argument("-Lz", type=float, default=2., help="Lz")
    parser.add_argument("-H", type=float, default=1., help="H")
    parser.add_argument("-R", type=float, default=1., help="R")
    parser.add_argument("-r", type=float, default=0.5, help="r")
    args = parser.parse_args()

    mesh = make_quarter_snoevsen_mesh(args.res, args.N, args.Lx,
                                      args.Ly, args.Lz,
                                      args.H, args.R, args.r)

    with df.XDMFFile("mesh.xdmf") as xdmff:
        xdmff.write(mesh)

    with df.HDF5File(mesh.mpi_comm(), "snoevsen_3d_quarter.h5", "w") as h5f:
        h5f.write(mesh, "mesh")


if __name__ == "__main__":
    main()
