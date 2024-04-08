#!/usr/bin/env python3

from mesh_sphere_packing.parse import Domain, Config
from mesh_sphere_packing.splitsphere import splitsphere
from mesh_sphere_packing.boundarypslg import boundarypslg
from mesh_sphere_packing.tetmesh import build_tetmesh

import numpy as np
from pore_mesh import xyz_shift
from check_rcp import fetch_geom
import argparse
import meshio
from meshtools.io import numpy_to_dolfin
import dolfin as df

def parse_args():
    parser = argparse.ArgumentParser(description="Make periodic mesh")
    parser.add_argument("infile", type=str, help="Input file")
    parser.add_argument("outfile", type=str, help="Output file")
    parser.add_argument("-R", type=float, default=0.5, help="R")
    #parser.add_argument("-Lx", type=float, default=10., help="Lx")
    #parser.add_argument("-Ly", type=float, default=10., help="Ly")
    #parser.add_argument("-Lz", type=float, default=10., help="Lz")
    parser.add_argument("-x_pad", type=float, default=0., help="Padding x")
    parser.add_argument("-x_shift", type=float, default=0., help="Shift x")
    parser.add_argument("-y_pad", type=float, default=0., help="Padding y")
    parser.add_argument("-y_shift", type=float, default=0., help="Shift y")
    parser.add_argument("-z_pad", type=float, default=0., help="Padding z")
    parser.add_argument("-z_shift", type=float, default=0., help="Shift z")
    parser.add_argument("-dx", type=float, default=0.1, help="Segment length")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    #shift = np.zeros(3)
    #shift[2] = args.z_shift
    shift = np.array([args.x_shift, args.y_shift, args.z_shift])

    L, R, x = fetch_geom(args.infile)

    R = args.R

    x = xyz_shift(x, shift, L)

    pad = np.array([args.x_pad, args.y_pad, args.z_pad])

    L[:] += 2*pad
    for d in range(3):
        x[:, d] += pad[d]

    PBC = np.ones(3, dtype=bool)

    domain = Domain(L, PBC)
    config = Config()

    config.segment_length = args.dx
    config.tetgen_max_volume = args.dx**3/2

    #x = np.loadtxt(args.infile)
    x = xyz_shift(x, shift, L)
    particles = np.hstack([np.arange(len(x)).reshape((-1, 1)), x, np.ones((len(x), 1))*R])

    sphere_pieces = splitsphere(domain, particles, config)

    #for sp in sphere_pieces:
    #    print(sp.sphere.x, sp.trans_flag, sp.sphere.x + sp.trans_flag * L) #.__dict__.keys())

    # Get accurate sphere centers and radii
    x_sph = np.array([[*(sphere_piece.sphere.x + sphere_piece.trans_flag * L), sphere_piece.sphere.r] for sphere_piece in sphere_pieces])

    #print(x_sph)

    # Check that old sphere centers match    
    x_dict = dict()
    for i, xi in enumerate(x_sph[:, :3]):
        key = tuple(xi)
        x_dict[key] = i

    for j, xi in enumerate(x):
        key = tuple(xi)
        #i = x_dict[key]
        assert( key in x_dict )
    print(x.shape, x_sph.shape)

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

    np.savetxt(f"{args.outfile}.obst", x_sph)

    with df.XDMFFile(mesh.mpi_comm(), f"{args.outfile}_show.xdmf") as xdmff:
        xdmff.write(mesh)