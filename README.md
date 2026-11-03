# meshtools

Various useful meshing tools.
## Mesh checks

Quality, edge sizes and periodicity of a tetrahedral mesh in XDMF with its arrays in HDF5
(as meshio and mesh-sphere-packing write it), the boundary triangles in a second file with
a `markers` attribute, a marker of 7 or more the solid surface. They need numpy, scipy,
numba, h5py, meshio and femtools, and neither dolfin nor matplotlib.

```python
from meshtools.check_quality import measure, report, failures
from meshtools.edge_sizes import measure as edge_sizes
from meshtools.check_periodicity import check

m = measure("mesh.xdmf", "mesh_facets.xdmf", h=0.1)
print("\n".join(report(m)))
print(failures(m))                          # the floors it falls below, if any
print("\n".join(edge_sizes("mesh.xdmf", "mesh_facets.xdmf", h=0.1, vmax=1e-4)))
check("mesh.xdmf", [2.0, 2.0, 0.3])         # periodic in the box [0, L]?
```

`femtools.xdmfread` reads such a file's datasets with their gzip chunks inflated on a
thread pool, and the connectivity a chunk-row at a time.

## Tests

```bash
python3 -m pytest tests
```
