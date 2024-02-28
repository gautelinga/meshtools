#!/usr/bin/env -S yade -x

from yade import pack
import numpy as np
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("outfile", type=str, help="Output file")
    parser.add_argument("-R", type=float, default=0.5, help="Radius")
    parser.add_argument("-Lx", type=float, default=10.0, help="Size x")
    parser.add_argument("-Ly", type=float, default=10.0, help="Size y")
    parser.add_argument("-Lz", type=float, default=10.0, help="Size z")
    parser.add_argument("-tol", type=float, default=1e-3, help="Tolerance")
    return parser.parse_args()

def map_back(x, L):
    if x < 0:
        x += L
    elif x >= L:
        x -= L
    return x

if __name__ == "__main__":
    args = parse_args()

    factor = 1.285
    tol = 1e-4
    Nmax = 1000

    L_ = np.array([args.Lx, args.Ly, args.Lz])
    L_ext = factor * L_

    data_ = np.zeros((Nmax, 2))

    for i in range(Nmax):
        rcp = pack.randomPeriPack(args.R, L_ext, rRelFuzz=0)
        size = rcp.cellSize.mean()
        measure = L_.mean()/size

        data_[i, :] = [factor, measure]
        if abs(measure - 1.0) < args.tol:
            break

        if i == 0:
            factor *= measure
        else:
            dd = data_[:i+1, :]
            popt = np.polyfit(dd[:, 0], dd[:, 1], 1)

            #plt.plot(dd[:, 0], dd[:, 1], '*')
            #plt.plot(dd[:, 0], popt[0]*dd[:, 0] + popt[1], 'k')
            #plt.show()

            factor = (1 - popt[1])/popt[0]

        L_ext = factor * L_
        print(i, L_ext, size, measure)
    
    print(i, L_ext, size, measure)

    dd = data_[:i+1, :]
    plt.plot(np.arange(i+1), dd[:, 0])
    plt.show()

    pos = np.zeros((len(rcp), 3))
    for i, (x, r) in enumerate(rcp):
        pos[i, :] = x/measure

    for d in range(3):
        print(pos[:, d].max())

    for i in range(len(pos)):
        for d in range(3):
            pos[i, d] = map_back(pos[i, d], L_[d])

    dmat = -np.ones((len(pos), len(pos)))
    for i in range(len(pos)):
        for j in range(i):
            dx = abs(pos[i, :] - pos[j, :])
            dx = np.minimum(dx, np.array(L_)-dx)
            dmat[i, j] = np.linalg.norm(dx)

    #plt.imshow(dmat)
    #plt.show()

    dist = sorted(dmat[dmat >= 0])
    #print(dist)
    plt.hist(dist, bins=256)
    plt.show()

    np.savetxt(args.outfile, pos)