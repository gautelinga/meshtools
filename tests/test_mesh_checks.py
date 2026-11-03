"""The mesh checks: `check_quality`'s measures on known shapes, its numba kernel against
the numpy definitions, `edge_sizes.unique_edges` against numpy's unique, `check_periodicity`
on a periodic mesh and on one with a node moved; and all of them import without dolfin or
matplotlib."""
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_imports_without_dolfin_or_matplotlib():
    script = ("import sys\n"
              "for name in ('dolfin', 'matplotlib', 'meshpy', 'skimage'):\n"
              "    sys.modules[name] = None\n"
              "import meshtools.check_quality, meshtools.check_periodicity\n"
              "import meshtools.edge_sizes\n"
              "print('OK')\n")
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, timeout=300)
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-2000:]


def test_the_package_import_loads_no_dependency():
    """`import meshtools` loads its own metadata only: every name loads on first use."""
    code = ("import sys, meshtools\n"
            "heavy = {'numpy', 'scipy', 'numba', 'h5py', 'meshio', 'mpi4py', 'dolfin',\n"
            "         'matplotlib', 'meshpy', 'skimage', 'femtools'}\n"
            "print(sorted(heavy & {m.split('.')[0] for m in sys.modules}))\n")
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, timeout=300)
    assert out.returncode == 0 and out.stdout.strip() == "[]", out.stdout + out.stderr[-2000:]


def test_quality_metrics_agree_with_known_shapes():
    from meshtools.check_quality import min_dihedral, tet_quality, tri_quality
    regular = np.array([[[1., 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]])
    assert abs(tet_quality(regular)[0][0] - 1. / np.sqrt(3.)) < 1e-12
    assert abs(min_dihedral(regular) - 70.528779) < 1e-5
    equilateral = np.array([[[0., 0, 0], [1, 0, 0], [0.5, np.sqrt(3.) / 2., 0]]])
    assert abs(tri_quality(equilateral)[0] - 1.) < 1e-12


def test_the_kernel_agrees_with_the_numpy_measures():
    from meshtools.check_quality import _tet, min_dihedral, tet_quality
    rng = np.random.default_rng(3)
    P = rng.random((300, 4, 3))
    P[:20, 3] = P[:20, :3].mean(axis=1) + 1e-3 * rng.random((20, 3))   # near flat
    X, T = P.reshape(-1, 3), np.arange(1200).reshape(-1, 4)
    q, _ = tet_quality(P)
    for i in range(len(P)):
        qk, _, lo, hi, _, _ = _tet(X, T[i])
        assert abs(qk - q[i]) < 1e-9 * max(1., q[i])
        assert abs(lo - min_dihedral(P[i:i + 1])) < 1e-6
        assert 0. < lo <= hi < 180.


def test_min_dihedral_reports_the_smallest_angle():
    from meshtools.check_quality import min_dihedral
    # a sliver: two nearly parallel faces meeting at a few degrees, other dihedrals near 90
    sliver = np.array([[[0., 0, 0], [1, 0, 0], [0.5, 0.02, 0], [0.5, 0.01, 0.5]]])
    assert min_dihedral(sliver) < 5., min_dihedral(sliver)


def _kuhn_box(n, L):
    """A periodic tet mesh of the box [0, L]^3: n^3 cubes, each cut into the six Kuhn tets
    (the same diagonal in every cube, so opposite faces match)."""
    g = np.linspace(0., L, n + 1)
    pts = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    idx = np.arange((n + 1) ** 3).reshape(n + 1, n + 1, n + 1)
    corner = idx[:-1, :-1, :-1].reshape(-1)
    step = np.array([(n + 1) ** 2, n + 1, 1])
    tets = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        path = [np.zeros(3, int)]
        for k in perm:
            path.append(path[-1] + np.eye(3, dtype=int)[k])
        tets.append(np.stack([corner + p @ step for p in path], 1))
    return pts, np.concatenate(tets).astype(np.int32)


def test_unique_edges_are_the_meshs_edges():
    # against numpy's row-wise unique over all six edges of every tet, vertex ids shuffled
    from meshtools.edge_sizes import unique_edges
    _, tets = _kuhn_box(5, 1.)
    perm = np.random.default_rng(6).permutation(tets.max() + 1).astype(np.int32)
    tets = perm[tets]
    E = unique_edges(tets, tets.max() + 1)
    ref = np.unique(np.sort(tets[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]]
                            .reshape(-1, 2), axis=1), axis=0)
    assert np.array_equal(E, ref)


def test_check_periodicity_passes_a_periodic_mesh_and_fails_a_moved_node(tmp_path, capsys):
    import meshio

    from meshtools.check_periodicity import check
    pts, tets = _kuhn_box(6, 2.)
    meshio.write(tmp_path / 'good.xdmf', meshio.Mesh(pts, [('tetra', tets)]))
    assert check(str(tmp_path / 'good.xdmf'), [2., 2., 2.])
    assert 'tris 72/72 congruent=True' in capsys.readouterr().out
    moved = pts.copy()
    k = np.flatnonzero((np.abs(moved[:, 0] - 2.) < 1e-12) & (moved[:, 1] > 0) & (moved[:, 1] < 2)
                       & (moved[:, 2] > 0) & (moved[:, 2] < 2))[0]
    moved[k, 1] += 1e-6                      # one node on the far x face, off its partner
    meshio.write(tmp_path / 'bad.xdmf', meshio.Mesh(moved, [('tetra', tets)]))
    assert not check(str(tmp_path / 'bad.xdmf'), [2., 2., 2.])
    assert 'axis x' in capsys.readouterr().out
