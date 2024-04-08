import numpy as np
import argparse
import matplotlib.pyplot as plt
from numba import jit
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze RCP")
    parser.add_argument("infile", type=str, help="Output file")
    parser.add_argument("-Rmax", type=float, default=5., help="Max distance")
    parser.add_argument("--finite", action="store_true", help="Not periodic boundaries")
    return parser.parse_args()

def fetch_geom(fname):
    with open(fname, "r") as infile:
        LR = np.array([float(Lstr) for Lstr in infile.readline().split("\n")[0].split(" ")])
        L_, R = LR[:3], LR[3]
        lines = infile.readlines()
        pos = np.zeros((len(lines), 3))
        for i, line in enumerate(lines):
            pos[i, :] = np.array([float(xstr) for xstr in line.split("\n")[0].split(" ")])
    return L_, R, pos

def get_exp_rdfs():
    refdir = "rdfs_brodin2024"
    dref = 42.5

    r = []
    g = []

    #fig, ax = plt.subplots(1, 1)
    for fname in os.listdir(refdir):
        if fname[:4] == "corr":
            dset = np.loadtxt(os.path.join(refdir, fname))
            ri = dset[:, 0]/dref
            gi = dset[:, 1]
            r.append(ri)
            g.append(gi)
            #ax.plot(ri, gi, label=fname)

    r = np.vstack(r).mean(axis=0)
    g = np.vstack(g).mean(axis=0)
    return r, g

@jit
def compute_dists(pos, L_, periodic=np.ones(3, dtype=bool)):
    dmat = -np.ones((len(pos), len(pos)))
    for i in range(len(pos)):
        if i % 100 == 0:
            print(i)
        for j in range(i):
            dx = np.zeros(3)
            for k in range(3):
                dx[k] = abs(pos[i, k] - pos[j, k])
                if periodic[k]:
                    dx[k] = min(dx[k], L_[k]-dx[k])
            dmat[i, j] = np.linalg.norm(dx)
    return dmat

if __name__ == "__main__":
    args = parse_args()

    #rexp, gexp = get_exp_rdfs()
    gexp, rexp = np.load("rdfs_brodin2024/test_gr.npy")
    rexp /= 42.5
    gexp /= gexp[-1]
    ids = rexp < args.Rmax
    rexp = rexp[ids]
    gexp = gexp[ids]

    #ax.plot(r, g, 'k--')
    #ax.legend()
    #plt.show()
    #exit()

    L_, R, pos = fetch_geom(args.infile)

    pbc = np.ones(3, dtype=bool)
    if args.finite:
        pbc[:] = False

    dmat = compute_dists(pos, L_, periodic=pbc).flatten()

    dist = dmat[np.logical_and(dmat > 0, dmat < min(args.Rmax, 0.5*L_.min()))]
    #dist = sorted(dmat[np.logical_and(dmat > 0, dmat < 0.5*L_.min())].flatten())
    
    print("done")
    w = 1./ (4 * np.pi * dist**2)

    h, r_bins = np.histogram(dist, bins=200, density=False, weights=w)

    fig, ax = plt.subplots(1, 1)
    rmid = 0.5*(r_bins[1:]+r_bins[:-1])
    dr = r_bins[1]-r_bins[0]

    rho = len(pos) / L_.prod()
    phi = 1 - rho * 4 * np.pi * R**3 / 3
    print(rho, phi)
    #denom = 4 * np.pi*rmid**2 * dr * rho * len(pos) / 2
    denom = dr * rho * len(pos) / 2

    ax.plot(rmid, h / denom, '.-', label="DEM simulations")
    ax.plot(rmid, np.ones_like(rmid), "--", label="Asymptotic theory")
    ax.plot(rexp, gexp, 'g-.', label="Brodin et al. experiment")

    ax.set_ylabel("RDF $g(r)$")
    ax.set_xlabel("distance $r / d$")
    #ax.semilogy()
    ax.set_xlim(0, None)
    #ax.set_ylim(3e-1, 1e1)
    ax.set_ylim(0, 10)
    ax.legend()

    plt.show()