#!/usr/bin/env python3

from mesh_sphere_packing.parse import Domain, Config
from mesh_sphere_packing.splitsphere import splitsphere
from mesh_sphere_packing.boundarypslg import boundarypslg
from mesh_sphere_packing.tetmesh import build_tetmesh

import numpy as np
from pore_mesh import xyz_shift
import argparse
import meshio
from meshtools.io import numpy_to_dolfin
import dolfin as df

def parse_args():
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("infile", type=str, help="Input file")
    parser.add_argument("outfile", type=str, help="Output file")
    parser.add_argument("-R", type=float, default=0.5, help="R")
    parser.add_argument("-Lx", type=float, default=10., help="Lx")
    parser.add_argument("-Ly", type=float, default=10., help="Ly")
    parser.add_argument("-Lz", type=float, default=10., help="Lz")
    parser.add_argument("-dx", type=float, default=0.1, help="Segment length")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    shift = np.zeros(3)

    L = np.array([args.Lx, args.Ly, args.Lz])
    PBC = np.ones(3, dtype=bool)

    domain = Domain(L, PBC)
    config = Config()
    config.segment_length = args.dx
    config.tetgen_max_volume = args.dx**3/2

    x = np.loadtxt(args.infile)
    x = xyz_shift(x, shift, L)
    particles = np.hstack([np.arange(len(x)).reshape((-1, 1)), x, np.ones((len(x), 1))*args.R])

    sphere_pieces = splitsphere(domain, particles, config)
    boundaries = boundarypslg(domain, sphere_pieces, config)
    mesh = build_tetmesh(domain, sphere_pieces, boundaries, config)

    vtkfile = f"{args.outfile}.vtk"
    mesh.write_vtk(vtkfile)
    m = meshio.read(vtkfile)

    nodes = m.points
    cells = [c for c in m.cells if c.type == "tetra"]

    elems = []
    for cellset in cells:
        elems.append(cellset.data)
    elems = np.vstack(elems)

    mesh = numpy_to_dolfin(nodes, elems)
    with df.HDF5File(mesh.mpi_comm(), f"{args.outfile}.h5", "w") as h5f:
        h5f.write(mesh, "mesh")

    with df.XDMFFile(mesh.mpi_comm(), f"{args.outfile}_show.xdmf") as xdmff:
        xdmff.write(mesh)