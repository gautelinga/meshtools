import numpy as np
import dolfin as df
import h5py
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# mesh_type = ["vertex", "line", "quad", "hexahedron"]

def remove_safe(path):
    """ Remove file in a safe way. """
    if rank == 0 and os.path.exists(path):
        os.remove(path)


def numpy_to_dolfin(nodes, elements, delete_tmp=True):
    """ Convert nodes and elements to a dolfin mesh object. """
    tmpfile = "tmp.h5"

    dim = nodes.shape[1]
    npts = elements.shape[1]
    if dim == 0:
        celltype_str = 'point'
    elif dim == 1:
        celltype_str = 'interval'
    elif dim == 2:
        if npts == 3:
            celltype_str = 'triangle'
        elif npts == 4:
            celltype_str = 'quadrilateral'
    elif dim == 3:
        if npts == 4:
            celltype_str = 'tetrahedron'
        elif npts == 8:
            celltype_str = 'hexahedron'
    
    if rank == 0:
        with h5py.File(tmpfile, "w") as h5f:
            cell_indices = h5f.create_dataset(
                "mesh/cell_indices", data=np.arange(len(elements)),
                dtype='int64')
            topology = h5f.create_dataset(
                "mesh/topology", data=elements, dtype='int64')
            coordinates = h5f.create_dataset(
                "mesh/coordinates", data=nodes, dtype='float64')
            topology.attrs["celltype"] = np.string_(celltype_str)
            topology.attrs["partition"] = np.array([0], dtype='uint64')

    comm.Barrier()

    mesh = df.Mesh()
    h5f = df.HDF5File(mesh.mpi_comm(), tmpfile, "r")
    h5f.read(mesh, "mesh", False)
    h5f.close()

    comm.Barrier()
    if delete_tmp:
        remove_safe(tmpfile)
    return mesh
