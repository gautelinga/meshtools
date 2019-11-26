import numpy as np
from meshtools import grid, smoothed_crossing_pipes, marching_cubes, \
    patch_up, remesh_surface, mesh_volume, numpy_to_dolfin, plot_mesh, \
    clean_mesh
from meshtools.voxels import (
    refine_voxels, get_subcluster, extract_backbone,
    voxels_to_dolfin, laplacian_filter)
import dolfin as df
import argparse
import pickle
from examples.helpers import str2num
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def pad(S, m=1):
    nx, ny, nz = S.shape
    S_new = np.ones((nx+2*m, ny+2*m, nz+2*m), dtype=S.dtype)*S.min()
    if m > 0:
        S_new[m:-m, m:-m, m:-m] = S
    elif m < 0:
        mm = -m
        S_new[:, :, :] = S[mm:-mm, mm:-mm, mm:-mm]
    else:
        S_new[:, :, :] = S
    return S_new


def extract_edges(faces):
    edges = set()
    for face in faces:
        n = len(face)
        for i in range(n-1):
            for j in range(i+1, n):
                edge = (min(face[i], face[j]),
                        max(face[i], face[j]))
                if edge in edges:
                    edges.remove(edge)
                else:
                    edges.add(edge)
    return edges


def make_v_dict(edges):
    v_dict = dict()
    for v1, v2 in edges:
        if v1 in v_dict:
            v_dict[v1].append(v2)
        else:
            v_dict[v1] = [v2]
        if v2 in v_dict:
            v_dict[v2].append(v1)
        else:
            v_dict[v2] = [v1]
    return v_dict


def closed_loops(edges):
    edges = list(edges)

    v_dict = make_v_dict(edges)

    v_stack = set(list(np.unique(edges)))

    v_lists = []
    while len(v_stack) > 0:
        v0 = v_stack.pop()
        v_prev = v0
        if v_dict[v0][0] in v_stack:
            v = v_dict[v0][0]
        elif v_dict[v0][1] in v_stack:
            v = v_dict[v0][1]
        else:
            continue
        v_stack.remove(v)

        v_list = [v0, v]
        while v != v0:
            if v_dict[v][0] != v_prev:
                v_next = v_dict[v][0]
            else:
                v_next = v_dict[v][1]
            v_list.append(v_next)
            v_prev = v
            v = v_next
            if v in v_stack:
                v_stack.remove(v)
            else:
                break
        v_lists.append(np.array(v_list))
    return v_lists


def open_segments(edges):
    edges = list(edges)
    v_dict = make_v_dict(edges)
    v_ends = set()
    for key, val in v_dict.items():
        if len(val) == 1:
            v_ends.add(key)

    v_lists = []
    while len(v_ends) > 0:
        v = v_ends.pop()
        v_next = v_dict[v][0]
        v_list = [v, v_next]
        while True:
            v_prev = v
            v = v_next
            if v_dict[v][0] != v_prev:
                v_next = v_dict[v][0]
            else:
                v_next = v_dict[v][1]
            v_list.append(v_next)
            if v_next in v_ends:
                v_ends.remove(v_next)
                break
        v_lists.append(np.array(v_list))
    return v_lists


parser = argparse.ArgumentParser(description="Fbleu-->mesh")
parser.add_argument("infile", type=str, help="Input file")
parser.add_argument("outfile", type=str, help="Output file")
parser.add_argument("--check", action="store_true",
                    help="Check for backbone.")
parser.add_argument("--refine", type=int, default=1,
                    help="Refine mesh.")
parser.add_argument("-D", type=str, default="(100,100,100)",
                    help="Subsample size")
parser.add_argument("-d", type=str, default="(40,40,40)",
                    help="Subsample displacement.")
args = parser.parse_args()


pkl_file = open(args.infile, "rb")
cluster = pickle.load(pkl_file)
bw = cluster > 0

D = str2num(args.D)
d = str2num(args.d)

print("Size of data:", cluster.shape)
print(D, d)

bw = get_subcluster(bw, D, d)

if args.check:
    bw = extract_backbone(bw)

# mesh, cells, nodes, cell_coords = voxels_to_dolfin(bw)
# with df.XDMFFile(mesh.mpi_comm(),
#                  "{}_vox.xdmf".format(args.outfile)) as xdmff:
#     xdmff.write(mesh)

print("Refining")
bw = refine_voxels(bw, args.refine)

S = np.zeros(bw.shape)
S[:, :, :] = 2.0*bw-1.0

print("Filtering")
for i in range(3):
    S = laplacian_filter(S, 0.1)

S = pad(S)

x_min, x_max = 0., 1.
y_min, y_max = 0., 1.
z_min, z_max = 0., 1.
bounding_box = np.array([[x_min, x_max],
                         [y_min, y_max],
                         [z_min, z_max]])

X, Y, Z = grid(bounding_box, S.shape)

dx = (x_max-x_min)/bw.shape[0]
dy = (y_max-y_min)/bw.shape[1]
dz = (z_max-z_min)/bw.shape[2]

bounding_box_extra = bounding_box + np.array([[-dx, dx],
                                              [-dy, dy],
                                              [-dz, dz]])

print("Marching cubes")
verts, faces = marching_cubes(S, bounding_box_extra)

print("Cleaning")
verts, faces = clean_mesh(verts, faces, bounding_box, [dx, dy, dz])


def roundit(r):
    N = 10000
    return float(np.round(r*N))/N


def roundvec(rv):
    return [roundit(r) for r in rv]


verts = np.array(
    [roundvec(vert) for vert in verts])

node_dict = dict()
old2pos = dict()
for i, vert in enumerate(verts):
    old2pos[i] = vert
    if vert in node_dict:
        node_dict[tuple(vert)].append(i)
    else:
        node_dict[tuple(vert)] = [i]




# print(node_dict)

# for i in range(len(faces)):
#     for j in range(3):
#         faces[i, j] = node_dict[tuple(verts[j, :])]
# verts = np.array(nodes)
# print(verts)

face_set = set()
for face in faces:
    face_set.add(tuple(sorted(face)))
faces = np.array(list(face_set))

import meshio
# cells = dict(triangle=np.array(faces))
# mesh = meshio.Mesh(verts, cells)

# import pygalmesh
# print("orienting")
# mesh = pygalmesh.orient_surface_mesh(mesh)
# verts = mesh.points
# faces = mesh.cells["triangle"]

print("Exporting")
meshio.write_points_cells(
    "{}.vtk".format(args.outfile),
    verts,
    dict(triangle=faces),
    )

import h5py
with h5py.File(args.outfile + ".h5", "w") as h5f:
    h5f.create_dataset("surface/node", data=verts)
    h5f.create_dataset("surface/face", data=faces)
