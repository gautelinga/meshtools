"""Whether a tetrahedral mesh is periodic in x, y, z: boundary nodes and boundary
triangles on opposite faces of the box [0, L] must coincide under translation by L.

    ok = check("mesh.xdmf", L)          # prints each axis in turn

Any mesh file meshio can read works; the tets are what is checked.

Only the triangles lying in the six boundary planes matter here, and those can be
had without deduplicating every face in the mesh: a tetrahedron has positive volume,
so it cannot have four vertices in one plane, and a face with three vertices in a
boundary plane has nothing on the other side of it to be shared with. So the faces
on a plane are exactly the on-plane vertices of the tets with three of them there.

It is fast at any size. An XDMF file with its data in HDF5 is read with
`femtools.xdmfread` (gzip chunks inflated on a thread pool), and the tets are scanned a
chunk-row at a time by one parallel numba kernel that marks those with three or four
vertices in a boundary plane; only those few are kept. The nodes and triangles are
matched by sorting arrays rather than through Python dictionaries. Other files are read
by meshio."""
import os

import numba
import numpy as np
import meshio
from numba import njit, prange

from femtools import xdmfread

TOL = 1e-8


@njit(parallel=True, cache=True)
def _plane_codes(T, bits):
    """Per tet, which of the six boundary planes hold three of its vertices (bits 0-5) or
    all four (bits 8-13). `bits[v]`: the planes vertex v lies in, bit 2*axis (lo) or
    2*axis+1 (hi)."""
    out = np.zeros(T.shape[0], np.uint16)
    for i in prange(T.shape[0]):
        code = 0
        for b in range(6):
            c = 0
            for k in range(4):
                c += (bits[T[i, k]] >> b) & 1
            if c == 3:
                code |= 1 << b
            elif c == 4:
                code |= 1 << (b + 8)
        out[i] = code
    return out


def _read(fname):
    """(points, tets as a chunk-row iterator, the number of tets)."""
    try:
        it = xdmfread.items(fname)
    except (OSError, ValueError, AttributeError, SyntaxError):
        it = None   # not an XDMF file with its data in HDF5: meshio reads it, as before
    if it is None:
        m = meshio.read(fname)
        tets = m.cells_dict["tetra"]
        return m.points, iter([(0, tets)]), len(tets)
    threads = int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) or os.cpu_count()
    pool = xdmfread.pool(threads)
    pts = xdmfread.read(xdmfread.open_item(it, 'geometry'), pool)
    tset = xdmfread.open_item(it, 'topology')
    return pts, xdmfread.row_blocks(tset, pool, ahead=max(2, threads // 4)), tset.shape[0]


def _rows(a):
    """Rows of a 2-D array as one sortable key each."""
    a = np.ascontiguousarray(a)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).reshape(-1)


def check(fname, L):
    """Checks the mesh in `fname` for periodicity in the box [0, L], reporting each axis
    in turn. Returns whether it passed.
    """
    pts, blocks, ntets = _read(fname)
    print(f"{len(pts)} nodes, {ntets} tets")
    numba.set_num_threads(min(int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) or
                              os.cpu_count(), numba.config.NUMBA_NUM_THREADS))

    # An explicit tolerance, not np.isclose: its relative term makes the test against
    # the far face L=6 accept anything within 6e-5 of it, and a node a whisker off a
    # boundary plane is the very defect this is looking for.
    on = [[np.abs(pts[:, ax] - side) <= TOL for side in (0., L[ax])] for ax in range(3)]
    bits = np.zeros(len(pts), np.uint8)
    for ax in range(3):
        for s_, mask in enumerate(on[ax]):
            bits[mask] |= 1 << (2 * ax + s_)

    # the faces in each plane: the on-plane vertices of the tets with three there
    faces = [[] for _ in range(6)]
    flat = np.zeros(6, np.int64)
    for _, T in blocks:
        T = np.ascontiguousarray(T)
        code = _plane_codes(T, bits)
        hit = np.flatnonzero(code)
        for b in range(6):
            three = hit[(code[hit] >> b) & 1 == 1]
            if len(three):
                cand = T[three]
                faces[b].append(cand[(bits[cand] >> b) & 1 == 1].reshape(-1, 3))
            flat[b] += int(((code[hit] >> (b + 8)) & 1).sum())
    faces = [np.concatenate(f) if f else np.empty((0, 3), np.int64) for f in faces]

    ok = True
    for ax in range(3):
        ilo, ihi = np.flatnonzero(on[ax][0]), np.flatnonzero(on[ax][1])
        klo = np.round(np.delete(pts[ilo], ax, axis=1), 8)
        khi = np.round(np.delete(pts[ihi], ax, axis=1), 8)
        # Node bijection: the in-plane coordinates of the two faces, each set without
        # repeats, the same sets
        olo, ohi = np.lexsort(klo.T[::-1]), np.lexsort(khi.T[::-1])
        slo, shi = klo[olo], khi[ohi]
        unique = lambda k: len(k) < 2 or bool((np.diff(k, axis=0) != 0).any(axis=1).all())
        nodes_ok = (len(ilo) == len(ihi) and unique(slo) and unique(shi)
                    and np.array_equal(slo, shi))
        partner = None
        if nodes_ok:
            partner = np.full(len(pts), -1, np.int64)
            partner[ilo[olo]] = ihi[ohi]
        # Triangle congruence: faces on the lo plane, mapped via partner, must be the
        # faces on the hi plane
        flo, fhi = faces[2 * ax], faces[2 * ax + 1]
        tris_ok = nodes_ok and len(flo) == len(fhi)
        if tris_ok and len(flo):
            mapped = np.sort(partner[flo], axis=1)
            tris_ok = bool((mapped >= 0).all()) and np.array_equal(
                np.sort(_rows(mapped)), np.sort(_rows(np.sort(fhi, axis=1).astype(mapped.dtype))))
        flat_lo, flat_hi = int(flat[2 * ax]), int(flat[2 * ax + 1])
        if flat_lo or flat_hi:
            tris_ok = False
            print(f"axis {'xyz'[ax]}: {flat_lo + flat_hi} tets lie flat in a boundary plane")
        # Max positional mismatch
        if partner is not None and len(ilo):
            a, b = ilo[olo], ihi[ohi]
            err = np.abs(np.delete(pts[a] - pts[b], ax, axis=1)).max()
        else:
            err = np.inf
        status = "OK" if (nodes_ok and tris_ok) else "FAIL"
        ok &= nodes_ok and tris_ok
        print(f"axis {'xyz'[ax]}: nodes {len(ilo)}/{len(ihi)} bijective={nodes_ok}, "
              f"tris {len(flo)}/{len(fhi)} congruent={tris_ok}, max mismatch={err:.2e}  [{status}]")

    # Edge/corner sanity: nodes on multiple faces must map consistently (checked implicitly above).
    print("PERIODICITY TEST PASSED" if ok else "PERIODICITY TEST FAILED")
    return ok
