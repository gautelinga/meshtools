import numpy as np
import vox2mesh
import pickle
import matplotlib.pyplot as plt
import dolfin as df
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert sample to mesh")
    parser.add_argument("infile", type=str, help="Input file")
    parser.add_argument("outfile", type=str, help="Output file")
    args = parser.parse_args()
    
    pkl_file = open(args.infile, "rb")
    cluster = pickle.load(pkl_file)
    bw = cluster > 0
    iic = vox2mesh.extract_backbone(bw)
    mesh, cells, nodes, cell_coords = vox2mesh.generate(iic)

    h5f = df.HDF5File(mesh.mpi_comm(), args.outfile, "w")
    h5f.write(mesh, "mesh")
    h5f.close()
    
    xdmff = df.XDMFFile(mesh.mpi_comm(), "fbleu/mesh.xdmf")
    xdmff.write(mesh)
    xdmff.close() 
