import numpy as np
from meshtools import grid, smoothed_crossing_pipes, marching_cubes, \
    patch_up, remesh_surface, mesh_volume, numpy_to_dolfin
import dolfin as df


N = (50, 50, 50)
R_left = 0.2
R_right = 0.1
x_min, x_max = 0., 1.
y_min, y_max = 0., 1.
z_min, z_max = 0., 1.
bounding_box = np.array([[x_min, x_max],
                         [y_min, y_max],
                         [z_min, z_max]])

X, Y, Z = grid(bounding_box, N)

S = smoothed_crossing_pipes(X, Y, Z, R_left, R_right)

verts, faces = marching_cubes(S, bounding_box)

verts, faces = patch_up(verts, faces, bounding_box)

verts, faces = remesh_surface(verts, faces)

# plot_mesh(verts, faces)

verts, elems = mesh_volume(verts, faces)

mesh = numpy_to_dolfin(verts, elems)

with df.XDMFFile(mesh.mpi_comm(), "mesh_cgal.xdmf") as xdmff:
    xdmff.write(mesh)

with df.HDF5File(mesh.mpi_comm(), "mesh_cgal_df.h5", "w") as h5f:
    h5f.write(mesh, "mesh")
