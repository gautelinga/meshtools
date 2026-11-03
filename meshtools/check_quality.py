"""Element quality of a tetrahedral mesh, measured and held to a floor.

    m = measure("mesh.xdmf", "mesh_facets.xdmf", h)   # every number, as a dict
    print("\n".join(report(m)))
    bad = failures(m)                                  # the floors it falls below

The files are XDMF with their arrays in HDF5, as meshio and mesh-sphere-packing write
them: the tetrahedra, and the boundary triangles with a `markers` attribute, where a
marker of `solid` or more is the solid surface. `h` is the length the shortest edge is
reported in units of.

Reported, for the tetrahedra:

  q = 12 (3V)^(2/3) / (sqrt(3) sum_edges l^2)

which is 1/sqrt(3) = 0.5774 for a regular tetrahedron and 0 for a flat one. Also the
minimum dihedral angle, the shortest edge in units of h, and the quality of the
triangles on the solid surface, 4*sqrt(3)*A / sum l^2, which is 1 for an equilateral one.

What is measured and what is enforced are not the same list: the floors below are
far beneath what a good build gives, and the two numbers that describe the single
worst element -- min quality and min dihedral -- are reported only. See the comment
on the constants.

The report also gives the dihedral angles' distribution (the smallest and largest
angle's percentiles, and how many tets are within 5 / 10 / 15 degrees of flat), the
radius-edge ratio, the total volume, and, given the beads (`beads=(box, centres,
radii)`), how far the solid-surface vertices are from their spheres.

It is fast at any size: the files are read with `femtools.xdmfread` (gzip chunks
inflated on a thread pool) and every tetrahedron is measured by one numba kernel, in
parallel, a chunk-row of the file at a time. Distributions are kept as histograms, so
percentiles are to the bin (0.001 degree; 1e-4 in log10 of the radius-edge ratio);
minima, maxima, counts and sums are exact.
"""
import os

import numba
import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

from femtools import xdmfread

# What stokes-beadpack's bead-pack builds, any packing in any mode at either resolution
# its suite is run at, have comfortably cleared. Over the eighteen cases at h=0.1: mean
# quality 0.448 to 0.463 (it goes with the mode -- shrunk 0.463, welded 0.453, flattened
# 0.449), the tail below 3.3e-4 of the elements, mean surface quality 0.897 welded to
# 0.969 shrunk. The floors sit well under those, since they are here to catch a regression
# rather than to grade a mesh.
MIN_MEAN_Q = 0.40
MAX_TAIL_FRACTION = 2e-3   # fraction of elements below q = 0.1
# The same for a Hele-Shaw cell, whose triangulated walls score 0.94 to 0.98 whatever
# the gap.
MIN_SURFACE_MEAN_Q = 0.85

# The minimum dihedral angle and the minimum element quality are reported but not
# held to anything. They measure one element out of a quarter of a million, and that
# one element belongs to the packing rather than to the mesher: packing03 has a
# 0.07-degree sliver in it at h=0.1 in every opening mode, shrinking included, while
# packing01 in the same run has nothing under 2 degrees. A floor loose enough to
# admit the first is too loose to catch anything. The tail fraction is the number
# that moves when the mesher gets worse.

EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# The histograms: dihedral angles in bins of 0.001 degree; the radius-edge ratio in bins
# of 1e-4 in log10, from 10^-0.25 (a regular tet is 0.612) to 10^3.25.
A_BIN = 1e-3
R_LO, R_BIN, R_N = -0.25, 1e-4, 35000


def tet_quality(T):
    """Shape quality and volume of each tetrahedron of `T`, shaped (n, 4, 3)."""
    v = np.abs(np.einsum('ij,ij->i', np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]),
                         T[:, 3] - T[:, 0])) / 6.
    el2 = sum(np.sum((T[:, i] - T[:, j])**2, axis=1) for i, j in EDGES)
    return 12. * (3. * v)**(2. / 3.) / np.sqrt(3.) / el2, v


def min_dihedral(T):
    """The smallest dihedral angle in the mesh, in degrees.

    Two faces of a tetrahedron share an edge, and the dihedral along it is pi minus
    the angle between the faces' outward normals -- so the normals have to be
    oriented, not merely computed: the normal of the face opposite vertex i points
    away from vertex i. Any two of the four faces meet, so all six pairings are
    dihedrals, one per edge.
    """
    n = np.empty(T.shape[:1] + (4, 3))
    for i in range(4):
        a, b, c = [k for k in range(4) if k != i]
        f = np.cross(T[:, b] - T[:, a], T[:, c] - T[:, a])
        away = np.einsum('ij,ij->i', f, T[:, i] - T[:, a]) > 0.
        n[:, i] = np.where(away[:, None], -f, f)
    n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-300)
    worst = 180.
    for i in range(4):
        for j in range(i + 1, 4):
            # arccos decreases, so the smallest angle is at the largest cosine.
            cos = -np.einsum('ij,ij->i', n[:, i], n[:, j])
            worst = min(worst, np.degrees(np.arccos(np.clip(cos.max(), -1., 1.))))
    return worst


def tri_quality(P):
    """Shape quality of each triangle of `P`, shaped (n, 3, 3); 1 is equilateral."""
    area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    el2 = sum(np.sum((P[:, i] - P[:, j])**2, axis=1) for i, j in [(0, 1), (0, 2), (1, 2)])
    return 4. * np.sqrt(3.) * area / el2


@njit(cache=True, inline='always')
def _normal(ax, ay, az, bx, by, bz, cx, cy, cz, ox, oy, oz):
    """The unit normal of face abc, turned away from the fourth vertex o."""
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    if nx * (ox - ax) + ny * (oy - ay) + nz * (oz - az) > 0.:
        nx, ny, nz = -nx, -ny, -nz
    r = max(np.sqrt(nx * nx + ny * ny + nz * nz), 1e-300)
    return nx / r, ny / r, nz / r


@njit(cache=True, inline='always')
def _dihedral(n1, n2):
    c = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]
    return 180. - np.degrees(np.arccos(min(1., max(-1., c))))


@njit(cache=True)
def _tet(X, t):
    """One tetrahedron: shape quality, 6 x signed volume, smallest and largest dihedral,
    radius-edge ratio, shortest edge squared. Scalars throughout: at billions of
    elements an array allocated per element is most of the time."""
    p0x, p0y, p0z = X[t[0], 0], X[t[0], 1], X[t[0], 2]
    p1x, p1y, p1z = X[t[1], 0], X[t[1], 1], X[t[1], 2]
    p2x, p2y, p2z = X[t[2], 0], X[t[2], 1], X[t[2], 2]
    p3x, p3y, p3z = X[t[3], 0], X[t[3], 1], X[t[3], 2]
    f3 = _normal(p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z)
    f2 = _normal(p0x, p0y, p0z, p1x, p1y, p1z, p3x, p3y, p3z, p2x, p2y, p2z)
    f1 = _normal(p0x, p0y, p0z, p2x, p2y, p2z, p3x, p3y, p3z, p1x, p1y, p1z)
    f0 = _normal(p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z, p0x, p0y, p0z)
    d = (_dihedral(f0, f1), _dihedral(f0, f2), _dihedral(f0, f3), _dihedral(f1, f2),
         _dihedral(f1, f3), _dihedral(f2, f3))
    lo, hi = min(d), max(d)
    ax, ay, az = p1x - p0x, p1y - p0y, p1z - p0z
    bx, by, bz = p2x - p0x, p2y - p0y, p2z - p0z
    cx, cy, cz = p3x - p0x, p3y - p0y, p3z - p0z
    bcx, bcy, bcz = by * cz - bz * cy, bz * cx - bx * cz, bx * cy - by * cx
    cax, cay, caz = cy * az - cz * ay, cz * ax - cx * az, cx * ay - cy * ax
    abx, aby, abz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
    vol6 = ax * bcx + ay * bcy + az * bcz
    aa, bb, cc = ax * ax + ay * ay + az * az, bx * bx + by * by + bz * bz, cx * cx + cy * cy + cz * cz
    dx, dy, dz = p2x - p1x, p2y - p1y, p2z - p1z
    ex, ey, ez = p3x - p1x, p3y - p1y, p3z - p1z
    fx, fy, fz = p3x - p2x, p3y - p2y, p3z - p2z
    dd, ee, ff = dx * dx + dy * dy + dz * dz, ex * ex + ey * ey + ez * ez, fx * fx + fy * fy + fz * fz
    el2 = aa + bb + cc + dd + ee + ff
    q = 12. * (3. * abs(vol6) / 6.) ** (2. / 3.) / np.sqrt(3.) / el2
    nx = aa * bcx + bb * cax + cc * abx
    ny = aa * bcy + bb * cay + cc * aby
    nz = aa * bcz + bb * caz + cc * abz
    l2 = min(aa, bb, cc, dd, ee, ff)
    rer = np.sqrt(nx * nx + ny * ny + nz * nz) / (2. * abs(vol6)) / np.sqrt(l2) if vol6 != 0. else np.inf
    return q, vol6, lo, hi, rer, l2


@njit(parallel=True, cache=True)
def _block(X, T, nb, h_lo, h_hi, h_flat, h_r, red):
    """Measures the tetrahedra T, `nb` blocks in parallel, each block into its own row of
    the histograms and of `red`: q sum, q min, q < 0.1, |volume| min, |volume| sum,
    positive, negative, shortest edge^2, smallest dihedral, largest dihedral, largest
    radius-edge ratio."""
    n = T.shape[0]
    na, nh, nr = h_lo.shape[1], h_hi.shape[1], h_r.shape[1]
    for b in prange(nb):
        for i in range(b * n // nb, (b + 1) * n // nb):
            q, v6, lo, hi, rer, l2 = _tet(X, T[i])
            v = abs(v6) / 6.
            red[b, 0] += q
            red[b, 1] = min(red[b, 1], q)
            if q < 0.1:
                red[b, 2] += 1
            red[b, 3] = min(red[b, 3], v)
            red[b, 4] += v
            if v6 > 0.:
                red[b, 5] += 1
            elif v6 < 0.:
                red[b, 6] += 1
            red[b, 7] = min(red[b, 7], l2)
            red[b, 8] = min(red[b, 8], lo)
            red[b, 9] = max(red[b, 9], hi)
            red[b, 10] = max(red[b, 10], rer)
            h_lo[b, min(int(lo / 1e-3), na - 1)] += 1
            h_hi[b, min(int(hi / 1e-3), nh - 1)] += 1
            h_flat[b, min(int(min(lo, 180. - hi) / 1e-3), na - 1)] += 1
            k = nr - 1
            if rer > 0. and np.isfinite(rer):
                k = min(max(int((np.log10(rer) - R_LO) / R_BIN), 0), nr - 1)
            h_r[b, k] += 1


def _percentiles(hist, ps, lo, width):
    """Percentiles of a histogram, as the centres of the bins they fall in."""
    c = np.cumsum(hist)
    return [lo + (np.searchsorted(c, p * c[-1] / 100.) + 0.5) * width for p in ps]


def sphere_deviation(X, tri, markers, box, centres, radii, solid=7):
    """How far the solid-surface vertices are from their sphere, against the sag of a
    chord as long as the median surface edge; spheres in a periodic `box`."""
    if not len(centres):
        return {}
    c, r = np.asarray(centres, float), np.asarray(radii, float)
    L = np.asarray(box, float)
    solid = tri[markers >= solid]
    sv = np.unique(solid)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]) * L
    centres = (c[None] + shifts[:, None]).reshape(-1, 3)
    radii = np.tile(r, len(shifts))
    _, idx = cKDTree(centres).query(X[sv], k=4, workers=-1)
    dev = np.min(np.abs(np.linalg.norm(X[sv][:, None] - centres[idx], axis=2) - radii[idx]),
                 axis=1)
    h = np.median(np.linalg.norm(X[solid[:, 0]] - X[solid[:, 1]], axis=1))
    return {'sphere_dev': np.percentile(dev, [50, 90, 99, 100]),
            'chord_sag': h ** 2 / (8. * np.median(r)), 'surface_edge': h,
            'radius': np.median(r)}


def measure(tet_file, facet_file, h, threads=None, solid=7, beads=None):
    """Every quality number this module knows about, for the tetrahedra in `tet_file` and
    the boundary triangles in `facet_file`. `beads`: (box, centres, radii), or None.

    The tetrahedra are measured a chunk-row of the file at a time, by `_block` on all
    threads, so the footprint is the vertices and a few chunks, whatever the size.
    """
    threads = threads or int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) or os.cpu_count()
    numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    pool = xdmfread.pool(threads)
    it = xdmfread.items(tet_file)
    X = xdmfread.read(xdmfread.open_item(it, 'geometry'), pool)
    tets = xdmfread.open_item(it, 'topology')
    n = tets.shape[0]
    nb = 4 * threads
    h_lo = np.zeros((nb, round(90 / A_BIN) + 1), np.int64)
    h_hi = np.zeros((nb, round(180 / A_BIN) + 1), np.int64)
    h_flat = np.zeros_like(h_lo)
    h_r = np.zeros((nb, R_N), np.int64)
    red = np.zeros((nb, 11))
    red[:, [1, 3, 7, 8]] = np.inf
    for _, T in xdmfread.row_blocks(tets, pool, ahead=max(2, threads // 4)):
        _block(X, np.ascontiguousarray(T), nb, h_lo, h_hi, h_flat, h_r, red)
    h_lo, h_hi, h_flat, h_r = (h.sum(axis=0) for h in (h_lo, h_hi, h_flat, h_r))
    n_pos, n_neg = int(red[:, 5].sum()), int(red[:, 6].sum())
    flat = np.cumsum(h_flat)

    fi = xdmfread.items(facet_file)
    FX = xdmfread.read(xdmfread.open_item(fi, 'geometry'), pool)
    tri = xdmfread.read(xdmfread.open_item(fi, 'topology'), pool)
    markers = xdmfread.read(xdmfread.open_item(fi, 'markers'), pool).reshape(-1)
    pool.shutdown()
    qs = tri_quality(FX[tri[markers >= solid]])
    # a mesh with no solid surface at all reports nan, not a traceback
    surface = dict(surface_tris=len(qs),
                   surface_mean_q=float(qs.mean()) if len(qs) else float('nan'),
                   surface_min_q=float(qs.min()) if len(qs) else float('nan'))

    return dict(
        h=h, tets=n, mean_q=float(red[:, 0].sum()) / max(n, 1),
        min_q=float(red[:, 1].min()), tail=int(red[:, 2].sum()),
        min_vol=float(red[:, 3].min()), volume=float(red[:, 4].sum()),
        inverted=min(n_pos, n_neg), positive=n_pos, negative=n_neg,
        min_dihedral=float(red[:, 8].min()), max_dihedral=float(red[:, 9].max()),
        shortest_edge=float(np.sqrt(red[:, 7].min())) / h,
        dihedral_lo_pct=_percentiles(h_lo, [0.01, 0.1, 1, 50], 0., A_BIN),
        dihedral_hi_pct=_percentiles(h_hi, [99.99, 99.9, 99, 50], 0., A_BIN),
        near_flat=[int(flat[round(t / A_BIN) - 1]) for t in (5, 10, 15)],
        max_radius_edge=float(red[:, 10].max()),
        radius_edge_pct=[10 ** y for y in _percentiles(h_r, [99.99, 99.9, 99, 50], R_LO, R_BIN)],
        **surface,
        **(sphere_deviation(FX, tri, markers, *beads, solid=solid) if beads is not None
           else {}),
    )


def report(m):
    """The lines of the report on `measure`'s numbers."""
    out = [f"{m['tets']} tets at h={m['h']:g}: quality mean {m['mean_q']:.4f} "
           f"min {m['min_q']:.4f}, {m['tail']} below 0.1 "
           f"({m['tail'] / m['tets']:.2e}), min dihedral {m['min_dihedral']:.2f} deg, "
           f"shortest edge {m['shortest_edge']:.3f} h"]
    if m['surface_tris']:
        out.append(f"{m['surface_tris']} solid-surface triangles: quality mean "
                   f"{m['surface_mean_q']:.4f} min {m['surface_min_q']:.4f}")
    else:
        out.append('no solid surface in this mesh, so there is no surface quality to report')
    out += ['smallest dihedral (deg), min / 0.01% / 0.1% / 1% / median: '
            f"{m['min_dihedral']:.3g} / " + ' / '.join(f'{y:.3g}' for y in m['dihedral_lo_pct']),
            'largest dihedral (deg), max / 99.99% / 99.9% / 99% / median: '
            f"{m['max_dihedral']:.4g} / " + ' / '.join(f'{y:.4g}' for y in m['dihedral_hi_pct']),
            'tets within 5 / 10 / 15 deg of flat (min(smallest, 180 - largest)): '
            + ' / '.join(str(k) for k in m['near_flat']),
            'radius-edge ratio, max / 99.99% / 99.9% / 99% / median: '
            f"{m['max_radius_edge']:.3g} / "
            + ' / '.join(f'{y:.3g}' for y in m['radius_edge_pct']),
            f"volume {m['volume']:.12g}; orientation: {m['positive']} positive, "
            f"{m['negative']} negative, {m['tets'] - m['positive'] - m['negative']} zero"]
    if 'sphere_dev' in m:
        out += ['bead-surface vertices off their sphere (|dist - R|), median / p90 / p99 / max: '
                + ' / '.join(f'{y:.2e}' for y in m['sphere_dev']),
                f"for scale: a chord of length h sags h^2/(8R) = {m['chord_sag']:.2e} "
                f"(h = median surface edge {m['surface_edge']:.4f}, R median {m['radius']:.3f})"]
    return out


def failures(m):
    """What in `measure`'s numbers falls below the floors, one line each; empty if none."""
    bad = []
    if m['min_vol'] <= 0.:
        bad.append(f"{m['min_vol']:.3e} is a flat element")
    if m['inverted']:
        bad.append(f"{m['inverted']} elements are oriented against the rest (inverted)")
    if m['mean_q'] < MIN_MEAN_Q:
        bad.append(f"mean quality {m['mean_q']:.4f} below {MIN_MEAN_Q}")
    if m['tail'] > MAX_TAIL_FRACTION * m['tets']:
        bad.append(f"{m['tail']} elements below q=0.1, over {MAX_TAIL_FRACTION:.0e} "
                   f"of {m['tets']}")
    if m['surface_tris'] and m['surface_mean_q'] < MIN_SURFACE_MEAN_Q:
        bad.append(f"mean surface triangle quality {m['surface_mean_q']:.4f} below "
                   f"{MIN_SURFACE_MEAN_Q}")
    return bad
