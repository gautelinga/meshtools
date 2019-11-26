from meshpy import triangle as tri
import numpy as np
from skimage import measure
import pygalmesh
import meshio


def mesh_in_polygon(x, y, allow_boundary_steiner=False):
    pts = list(zip(x, y))
    edgs = list(zip(list(range(len(pts))),
                    list(range(1, len(pts))) + [0]))
    mi = tri.MeshInfo()
    mi.set_points(pts)
    mi.set_facets(edgs)

    ds = np.mean([np.linalg.norm(np.array(pts[i])-np.array(pts[j]))
                  for i, j in edgs])
    max_vol = np.sqrt(3)/4*ds*ds

    mesh = tri.build(mi, max_volume=max_vol, min_angle=25,
                     allow_boundary_steiner=allow_boundary_steiner)
    node = np.array(mesh.points)
    face = np.array(mesh.elements)
    return node, face


def merge_surfaces(surfaces, M=100000):
    node = []
    face = []
    n = 0
    for node_loc, face_loc in surfaces:
        node.append(node_loc)
        face.append(face_loc + n)
        n += np.size(node_loc, 0)
    node = np.vstack(node)
    face = np.vstack(face)
    xdict = dict()
    to_unique = dict()
    count_unique = 0
    for i, x in enumerate(node.tolist()):
        tup = tuple((M*np.array(x)).astype(int)/float(M))
        if tup not in xdict:
            xdict[tup] = count_unique
            to_unique[i] = count_unique
            count_unique += 1
        else:
            to_unique[i] = xdict[tup]
    node_new = np.zeros((count_unique, 3), dtype=float)
    for key, i in xdict.items():
        node_new[i, :] = np.array(key)
    face_new = np.array([[to_unique[f[i]] for i in range(3)]
                         for f in face.tolist()], dtype=int)
    return node_new, face_new


def find_plane_edges(verts, faces, dim, pos):
    ids_side = np.argwhere(np.abs(verts[:, dim] - pos) < 1e-5)
    print(ids_side)

    import matplotlib.pyplot as plt
    dims = list(range(3))
    dims.pop(dim)
    plt.scatter(verts[ids_side, dims[0]], verts[ids_side, dims[1]])
    plt.show()

    side_edges = dict()
    for face in faces:
        ids_on_side = np.intersect1d(face, ids_side)
        if len(ids_on_side) == 2:
            if ids_on_side[0] in side_edges:
                side_edges[ids_on_side[0]][1] = ids_on_side[1]
            else:
                side_edges[ids_on_side[0]] = [ids_on_side[1], None]
            if ids_on_side[1] in side_edges:
                side_edges[ids_on_side[1]][1] = ids_on_side[0]
            else:
                side_edges[ids_on_side[1]] = [ids_on_side[0], None]
    keys = list(side_edges.keys())
    for key in keys:
        if side_edges[key][1] is None:
            side_edges.pop(key)
    return side_edges


def _get_next(id_curr, id_prev, edges):
    for i in edges[id_curr]:
        if i != id_prev:
            return i


def compute_edge_lists(side_edges):
    edge_lists = []
    while len(side_edges):
        ids_first = list(side_edges.items())[0]
        id_first = ids_first[0]

        id_prev = id_first
        id_curr = ids_first[1][0]
        id_next = _get_next(id_curr, id_first, side_edges)

        edge_list = [[id_first, id_curr],
                     [id_curr, id_next]]
        side_edges.pop(id_first)
        side_edges.pop(id_curr)

        while id_next != id_first:
            id_prev = id_curr
            id_curr = id_next
            id_next = _get_next(id_curr, id_prev, side_edges)
            side_edges.pop(id_curr)
            edge_list.append([id_curr, id_next])

        edge_lists.append(edge_list)
    return edge_lists


def make_patch(verts, edge_list, dim, pos):
    other_dims = list(range(3))
    other_dims.remove(dim)
    ids = np.array(edge_list)[:, 0]
    xy = verts[ids, :]
    xy = xy[:, other_dims]
    x = xy[:, 0]
    y = xy[:, 1]
    verts_2d, faces_2d = mesh_in_polygon(x, y)
    # plot_2d(verts_2d, faces_2d)
    new_verts = np.zeros((len(verts_2d[:, 0]), 3))
    for i, d in enumerate(other_dims):
        new_verts[:, d] = verts_2d[:, i]
    new_verts[:, dim] = pos
    return (new_verts, faces_2d)


def patch_up(verts, faces, bounding_box):
    mesh_patches = [(verts, faces)]
    for dim, bounds in enumerate(bounding_box):
        for pos in bounds:
            side_edges = find_plane_edges(verts, faces, dim, pos)
            print(side_edges)
            edge_lists = compute_edge_lists(side_edges)
            for edge_list in edge_lists:
                patch = make_patch(verts, edge_list, dim, pos)
                mesh_patches.append(patch)

    verts, faces = merge_surfaces(mesh_patches)
    return verts, faces


def polygon_area(points):
    """Return the area of the polygon whose vertices are given by the
    sequence points.

    """
    area = 0
    q = points[-1]
    for p in points:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return area / 2


def marching_cubes(S, bounding_box):
    N = S.shape
    verts, faces, normals, values = measure.marching_cubes_lewiner(S, level=0.)

    for dim in range(3):
        verts[:, dim] = (bounding_box[dim, 0] +
                         (bounding_box[dim, 1] - bounding_box[dim, 0])
                         * verts[:, dim]/(N[dim]-1))

    return verts, faces


def clean_mesh(verts, faces, bounding_box, dX):
    dx, dy, dz = dX
    x_min, x_max = bounding_box[0, :]
    y_min, y_max = bounding_box[1, :]
    z_min, z_max = bounding_box[2, :]

    # Clean mesh
    verts[verts[:, 0] < x_min + dx/2, 0] = x_min
    verts[verts[:, 0] > x_max - dx/2, 0] = x_max
    verts[verts[:, 1] < y_min + dy/2, 1] = y_min
    verts[verts[:, 1] > y_max - dy/2, 1] = y_max
    verts[verts[:, 2] < z_min + dz/2, 2] = z_min
    verts[verts[:, 2] > z_max - dz/2, 2] = z_max

    point_dict = dict()
    for i, vert in enumerate(verts):
        x, y, z = vert
        if (x, y, z) in point_dict:
            point_dict[(x, y, z)].append(i)
        else:
            point_dict[(x, y, z)] = [i]

    for key, vals in point_dict.items():
        if len(vals) > 1:
            for val in vals[1:]:
                faces[faces[:, :] == val] = vals[0]

    bad_ids = set()
    for j, face in enumerate(faces):
        a, b, c = face
        if (a == b) or (a == c) or (b == c):
            bad_ids.add(j)

    good_ids = list(set(range(len(faces)))-bad_ids)
    faces = faces[good_ids, :]
    return verts, faces


def remesh_surface(verts, faces,
                   edge_size=0.025,
                   facet_angle=25.0,
                   facet_size=0.025,
                   facet_distance=0.025,
                   verbose=True):
    cells = dict(triangle=np.array(faces))
    mesh = meshio.Mesh(verts, cells)
    if verbose:
        print("orienting")
    mesh = pygalmesh.orient_surface_mesh(mesh)
    if verbose:
        print("remeshing")
    mesh = pygalmesh.remesh_surface(mesh,
                                    edge_size=edge_size,
                                    facet_angle=edge_size,
                                    facet_size=facet_size,
                                    facet_distance=facet_distance,
                                    verbose=True)

    if verbose:
        print("orienting again")
    mesh = pygalmesh.orient_surface_mesh(mesh)
    verts = mesh.points
    faces = mesh.cells["triangle"]
    return verts, faces
