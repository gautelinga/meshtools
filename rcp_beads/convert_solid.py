import meshio
mesh = meshio.read("porous_example_solid.msh")

nodes = mesh.points
elems = mesh.cells_dict["tetra"]

import dolfin as df
import meshtools
dfmesh = meshtools.io.numpy_to_dolfin(nodes, elems)

with df.XDMFFile(dfmesh.mpi_comm(), "porous_example_solid_2.xdmf") as xdmff:
    xdmff.write(dfmesh)
