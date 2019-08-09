import argparse
import matplotlib.pyplot as plt
import dolfin as df
from meshtools.voxel_fields import percolation_cluster
from meshtools.voxels import voxels_to_dolfin
from meshtools.plot import plot_vox, plot_voxel_mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make a periodic voxel structure.")
    parser.add_argument("-N", type=int, default=16,
                        help="Number of voxels")
    parser.add_argument("-D", "--dim", type=int, default=3,
                        help="Dimensions")
    parser.add_argument("-A", "--axis", type=int, default=0,
                        help="Axis")
    parser.add_argument("--plot", action="store_true", help="Plot")
    parser.add_argument("--xdmf", action="store_true", help="Store XDMF")
    parser.add_argument("-o", "--outfile", type=str, default="",
                        help="Outfile")
    parser.add_argument("-p", type=float, default=0.35,
                        help="Percolation probability")
    args = parser.parse_args()

    iic = percolation_cluster(args.N, args.dim, args.p, axis=args.axis)

    mesh, nodes, cells, cell_coords = voxels_to_dolfin(iic)

    if args.outfile is not "":
        fname = args.outfile.split(".")
        ext = fname[-1]
        if ext in ["h5", "hdf", "hdf5"]:
            h5f = df.HDF5File(mesh.mpi_comm(), args.outfile, "w")
            h5f.write(mesh, "mesh")
            h5f.close()

    if args.xdmf:
        xdmff = df.XDMFFile(mesh.mpi_comm(), "mesh.xdmf")
        xdmff.write(mesh)
        xdmff.close()

    if args.plot:
        plot_vox(nodes*args.N, cells, cell_coords, iic)
        plot_voxel_mesh(nodes, cells)
        plt.show()
