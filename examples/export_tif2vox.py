import argparse
import matplotlib.pyplot as plt
import pickle
from meshtools.voxels import tif2vox


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert tif data to voxels.")
    parser.add_argument("infile", type=str, help="Input file")
    parser.add_argument("outfile", type=str, help="Output file")
    parser.add_argument("--plot", action="store_true", help="Do plot")
    args = parser.parse_args()

    A = tif2vox(args.infile)

    if ".pkl" in args.outfile:
        pkl_file = open(args.outfile, "wb")
        pickle.dump(A, pkl_file)
        pkl_file.close()
    else:
        print("Wrong extension on outfile")

    if args.plot:
        for d in range(3):
            plt.figure()
            plt.imshow(A.sum(axis=d))
        plt.show()
