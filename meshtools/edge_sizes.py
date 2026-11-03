"""Edge lengths against h, for the self-similarity of a mesh ladder.

    lines = measure("mesh.xdmf", "mesh_facets.xdmf", h, vmax)

The solid-surface edges' 10th, 50th and 90th percentiles in units of h; the volume edges'
the same, binned by the distance of the edge's midpoint from the nearest solid-surface
vertex (0-0.5, 0.5-1, 1-2, 2-4 and beyond 4 h); and the mean tet volume against the
mesher's bound `vmax`. A boundary triangle is on the solid surface where its marker is
`solid` or more.

Each vertex's higher-numbered neighbours are gathered and deduplicated in parallel
(numba), the k-d tree is queried on every thread, and the mesh is read with
`femtools.xdmfread`: the same edges, lengths and percentiles as numpy's row-wise unique
over all six edges of every tet, without its time and memory.
"""
import os

import numba
import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

from femtools import xdmfread

PAIRS = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
BINS = ((0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 99))


@njit(cache=True)
def _count(T, nv, counts):
    """counts[a] += 1 for every edge (a < b) of every tet, duplicates included."""
    for i in range(T.shape[0]):
        for p in range(6):
            a, b = T[i, PAIRS[p, 0]], T[i, PAIRS[p, 1]]
            counts[min(a, b)] += 1


@njit(cache=True)
def _fill(T, offsets, cursor, nbr):
    for i in range(T.shape[0]):
        for p in range(6):
            a, b = T[i, PAIRS[p, 0]], T[i, PAIRS[p, 1]]
            lo, hi = min(a, b), max(a, b)
            nbr[offsets[lo] + cursor[lo]] = hi
            cursor[lo] += 1


@njit(parallel=True, cache=True)
def _dedup(offsets, nbr, keep):
    """Sorts each vertex's neighbour list in place and counts its distinct entries."""
    nv = offsets.shape[0] - 1
    for v in prange(nv):
        s, e = offsets[v], offsets[v + 1]
        if e > s:
            seg = np.sort(nbr[s:e])
            nbr[s:e] = seg
            k = 1
            for j in range(1, e - s):
                if seg[j] != seg[j - 1]:
                    k += 1
            keep[v] = k


@njit(parallel=True, cache=True)
def _edges(offsets, nbr, start, out):
    """The distinct (a, b) pairs, a < b, in vertex order."""
    nv = offsets.shape[0] - 1
    for v in prange(nv):
        s, e = offsets[v], offsets[v + 1]
        k = start[v]
        for j in range(s, e):
            if j == s or nbr[j] != nbr[j - 1]:
                out[k, 0] = v
                out[k, 1] = nbr[j]
                k += 1


@njit(parallel=True, cache=True)
def _volume_sum(X, T, nb):
    n = T.shape[0]
    part = np.zeros(nb)
    for b in prange(nb):
        acc = 0.
        for i in range(b * n // nb, (b + 1) * n // nb):
            a, bb, c, e = T[i, 0], T[i, 1], T[i, 2], T[i, 3]
            ux, uy, uz = X[bb, 0] - X[a, 0], X[bb, 1] - X[a, 1], X[bb, 2] - X[a, 2]
            vx, vy, vz = X[c, 0] - X[a, 0], X[c, 1] - X[a, 1], X[c, 2] - X[a, 2]
            wx, wy, wz = X[e, 0] - X[a, 0], X[e, 1] - X[a, 1], X[e, 2] - X[a, 2]
            acc += abs(ux * (vy * wz - vz * wy) + uy * (vz * wx - vx * wz)
                       + uz * (vx * wy - vy * wx))
        part[b] = acc
    return part.sum() / 6.


def unique_edges(T, nv):
    """The mesh's distinct edges (a < b), from the tets `T` over `nv` vertices."""
    counts = np.zeros(nv, np.int64)
    _count(T, nv, counts)
    offsets = np.zeros(nv + 1, np.int64)
    np.cumsum(counts, out=offsets[1:])
    nbr = np.empty(offsets[-1], T.dtype)
    _fill(T, offsets, np.zeros(nv, np.int64), nbr)
    keep = np.zeros(nv, np.int64)
    _dedup(offsets, nbr, keep)
    start = np.zeros(nv + 1, np.int64)
    np.cumsum(keep, out=start[1:])
    E = np.empty((start[-1], 2), np.int64)
    _edges(offsets, nbr, start[:-1].copy(), E)
    return E


def measure(tet_file, facet_file, h, vmax, threads=None, solid=7, label=None):
    """The lines of the size report for the tetrahedra in `tet_file` and the boundary
    triangles in `facet_file`, headed by `label` (default the tetrahedra's file)."""
    label = tet_file if label is None else label
    threads = threads or int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) or os.cpu_count()
    numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    pool = xdmfread.pool(threads)
    it, fi = xdmfread.items(tet_file), xdmfread.items(facet_file)
    X = xdmfread.read(xdmfread.open_item(it, 'geometry'), pool)
    T = xdmfread.read(xdmfread.open_item(it, 'topology'), pool)
    tri = xdmfread.read(xdmfread.open_item(fi, 'topology'), pool)
    mark = xdmfread.read(xdmfread.open_item(fi, 'markers'), pool).reshape(-1)
    pool.shutdown()
    solid = tri[mark >= solid]

    E = unique_edges(np.ascontiguousarray(T), len(X))
    L = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1) / h
    s = np.sort(np.concatenate([solid[:, [a, b]] for a, b in ((0, 1), (0, 2), (1, 2))]), axis=1)
    s = s.astype(np.int64)
    S = np.unique(s[:, 0] * len(X) + s[:, 1])
    Ls = np.linalg.norm(X[S // len(X)] - X[S % len(X)], axis=1) / h
    vol_mean = _volume_sum(X, np.ascontiguousarray(T), 4 * threads) / len(T)
    mid = 0.5 * (X[E[:, 0]] + X[E[:, 1]])
    dist = cKDTree(X[np.unique(solid)]).query(mid, workers=threads)[0] / h
    del mid

    def q(a):
        return ' / '.join(f'{y:.2f}' for y in np.percentile(a, [10, 50, 90]))

    out = [f'== {label}: h {h:.4g}, {len(T)} tets, mean tet volume {vol_mean / vmax:.2f} x the bound',
           f'   bead-surface edges (p10/median/p90, /h): {q(Ls)}']
    for lo, hi in BINS:
        m = (dist >= lo) & (dist < hi)
        if m.sum():
            top = str(hi) if hi < 99 else 'inf'
            out.append(f'   volume edges, wall distance {lo:3.1f}-{top:<3} h: {q(L[m])}  '
                       f'({m.sum()} edges)')
    return out
