#!/usr/bin/env -S yade -x

#from yade import pack
from yade.pack import (_getMemoizedPacking, _memoizePacking, O, SpherePack, Vector3, 
                       utils, ForceResetter, InsertionSortCollider, Bo1_Sphere_Aabb,
                       InteractionLoop, Ig2_Sphere_Sphere_ScGeom, Ip2_FrictMat_FrictMat_FrictPhys,
                       Law2_ScGeom_FrictPhys_CundallStrack, PeriIsoCompressor, NewtonIntegrator,
                       FrictMat)
import numpy as np
import argparse
import matplotlib.pyplot as plt
#from pore_mesh import xyz_shift

def randomPeriPack2(radius, initSize, rRelFuzz=0.0, memoizeDb=None, noPrint=False, seed=-1):
    """Generate periodic dense packing.

    A cell of initSize is stuffed with as many spheres as possible, then we run periodic compression with PeriIsoCompressor, just like with
    randomDensePack.

    :param radius: mean sphere radius
    :param rRelFuzz: relative fuzz of sphere radius (equal distribution); see the same param for randomDensePack.
    :param initSize: initial size of the periodic cell.

    :return: SpherePack object, which also contains periodicity information.
    """
    from math import pi
    sp = _getMemoizedPacking(
            memoizeDb,
            radius,
            rRelFuzz,
            initSize[0],
            initSize[1],
            initSize[2],
            fullDim=Vector3(0, 0, 0),
            wantPeri=True,
            fillPeriodic=False,
            spheresInCell=-1,
            memoDbg=True,
            noPrint=noPrint
    )
    if sp:
        return sp
    O.switchScene()
    O.resetThisScene()

    sp = SpherePack()
    O.periodic = True
    #O.cell.refSize=initSize
    O.cell.setBox(initSize)
    sp.makeCloud(Vector3().Zero, O.cell.refSize, radius, rRelFuzz, -1, True, seed=seed)
    O.engines = [
            ForceResetter(),
            InsertionSortCollider([Bo1_Sphere_Aabb()], verletDist=.05 * radius),
            InteractionLoop([Ig2_Sphere_Sphere_ScGeom()], [Ip2_FrictMat_FrictMat_FrictPhys()], [Law2_ScGeom_FrictPhys_CundallStrack()]),
            PeriIsoCompressor(
                    charLen=2 * radius, stresses=[-100e9, -1e8], maxUnbalanced=1e-3, doneHook='O.pause();', globalUpdateInt=20, keepProportions=True
            ),
            NewtonIntegrator(damping=.8)
    ]
    O.materials.append(FrictMat(young=30e9, frictionAngle=.1, poisson=.3, density=1e3))
    for s in sp:
        O.bodies.append(utils.sphere(s[0], s[1]))
    O.dt = utils.PWaveTimeStep()
    O.timingEnabled = True
    O.run()
    O.wait()

    ret = SpherePack()
    ret.fromSimulation()
    _memoizePacking(memoizeDb, ret, radius, rRelFuzz, wantPeri=True, fullDim=Vector3(0, 0, 0), noPrint=noPrint)  # fullDim unused
    O.switchScene()
    return ret

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("outfile", type=str, help="Output file")
    parser.add_argument("-R", type=float, default=0.5, help="Radius")
    parser.add_argument("-Lx", type=float, default=10.0, help="Size x")
    parser.add_argument("-Ly", type=float, default=10.0, help="Size y")
    parser.add_argument("-Lz", type=float, default=10.0, help="Size z")
    parser.add_argument("-tol", type=float, default=1e-3, help="Tolerance")
    return parser.parse_args()

def xyz_shift(x, dx, L):
    assert(len(dx) == len(L))
    xnew = np.zeros_like(x)
    for d in range(len(L)):
        dx[d] = np.remainder(dx[d], L[d])
        xnew[:, d] = x[:, d] + dx[d]  # np.outer(np.ones(len(x)), dx[d])
        xnew[:, d] = np.remainder(xnew[:, d], L[d])
    return xnew

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
        #rcp = pack.randomPeriPack(args.R, L_ext, rRelFuzz=0)
        rcp = randomPeriPack2(args.R, L_ext, rRelFuzz=0)
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

    for i in range(len(pos)):
        for d in range(3):
            pos[i, d] = map_back(pos[i, d], L_[d])

    # pos = xyz_shift(pos, np.array([0., 0., 0.]), L_)

    for d in range(3):
        print(d, pos[:, d].min(), pos[:, d].max())

    with open(args.outfile, "w") as ofile:
        ofile.write(" ".join([f"{Li}" for Li in L_.tolist()]) + f" {args.R}\n")
        np.savetxt(ofile, pos)