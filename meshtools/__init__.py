import importlib

from .__about__ import (
    __author__,
    __author_email__,
    __copyright__,
    __license__,
    __version__,
    __maintainer__,
    __status__,
    )

# imported on first use, so a submodule that needs neither dolfin nor matplotlib
# (check_quality, check_periodicity, edge_sizes) imports without them
_FROM = {
    "remove_safe": "io",
    "generate_tetgen_input": "io",
    "save_mesh": "io",
    "numpy_to_dolfin": "io",
    "mpi_comm": "cmd",
    "mpi_barrier": "cmd",
    "mpi_rank": "cmd",
    "mpi_size": "cmd",
    "mpi_is_root": "cmd",
    "plot_mesh": "plot",
    "plot_2d": "plot",
    "find_plane_edges": "surface",
    "patch_up": "surface",
    "mesh_in_polygon": "surface",
    "merge_surfaces": "surface",
    "polygon_area": "surface",
    "marching_cubes": "surface",
    "remesh_surface": "surface",
    "clean_mesh": "surface",
    "mesh_volume": "volume",
    "grid": "voxels",
    "laplacian_filter": "voxels",
    "smoothed_crossing_pipes": "voxel_fields",
}


def __getattr__(name):
    if name in _FROM:
        value = getattr(importlib.import_module("." + _FROM[name], __name__), name)
    else:
        try:
            value = importlib.import_module("." + name, __name__)
        except ModuleNotFoundError as e:
            if e.name != __name__ + "." + name:
                raise
            raise AttributeError("module {!r} has no attribute {!r}".format(
                __name__, name)) from None
    globals()[name] = value
    return value


__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__license__",
    "__version__",
    "__maintainer__",
    "__status__",
    #
    "remove_safe",
    "generate_tetgen_input",
    "save_mesh",
    "numpy_to_dolfin",
    #
    "mpi_comm",
    "mpi_barrier",
    "mpi_rank",
    "mpi_size",
    "mpi_is_root",
    #
    "plot_mesh",
    "plot_2d",
    #
    "find_plane_edges",
    "patch_up",
    "mesh_in_polygon",
    "merge_surfaces",
    "polygon_area",
    "marching_cubes",
    "remesh_surface",
    #
    "mesh_volume",
    #
    "grid",
    "laplacian_filter",
    #
    "smoothed_crossing_pipes",
]
