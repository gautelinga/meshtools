import pickle
import dolfin as df
import argparse
from meshtools.voxels import (
    refine_voxels, get_subcluster, extract_backbone,
    voxels_to_dolfin)


def str2num(s):
    if s[0] == "(" and s[-1] == ")":
        # s is tuple
        return tuple([int(l) for l in s[1:-1].split(",")])
    else:
        # s is int
        return int(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert sample to mesh")
    parser.add_argument("infile", type=str, help="Input file")
    parser.add_argument("outfile", type=str, help="Output file")
    parser.add_argument("--nocheck", action="store_true",
                        help="Disable check for backbone.")
    parser.add_argument("--refine", type=int, default=1,
                        help="Refine mesh.")
    parser.add_argument("-D", type=str, default="(100,100,100)",
                        help="Subsample size")
    parser.add_argument("-d", type=str, default="(0,0,0)",
                        help="Subsample displacement.")
    args = parser.parse_args()

    pkl_file = open(args.infile, "rb")
    cluster = pickle.load(pkl_file)
    bw = cluster > 0

    D = str2num(args.D)
    d = str2num(args.d)

    print(D, d)

    bw = get_subcluster(bw, D, d)

    if not args.nocheck:
        bw = extract_backbone(bw)

    bw = refine_voxels(bw, args.refine)
    mesh, cells, nodes, cell_coords = voxels_to_dolfin(bw)

    with df.HDF5File(mesh.mpi_comm(), args.outfile, "w") as h5f:
        h5f.write(mesh, "mesh")

    with df.XDMFFile(mesh.mpi_comm(), "fbleu/mesh.xdmf") as xdmff:
        xdmff.write(mesh)
