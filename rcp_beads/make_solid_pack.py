import numpy as np

import argparse
import h5py
import meshio

from scipy.spatial.qhull import ConvexHull

import matplotlib.pyplot as plt

def periodize(data, L, R):
    data_added = []

    ids_ = [[None for dim in range(3)] for _ in range(2)]
    for dim in range(3):
        ids_[0][dim] = data[:, dim] < R
        ids_[1][dim] = data[:, dim] > L[dim] - R

    for l in range(2):
        for dim in range(3):
            data_loc = data[ids_[l][dim], :]
            data_loc[:, dim] += (-1)**l * L[dim]
            data_added.append(data_loc)

    for dim in range(3):
        odims = list(range(3))
        odims.remove(dim)
        for l in range(2):
            for k in range(2):
                ids_comb = np.logical_and(ids_[l][odims[0]], ids_[k][odims[1]])
                data_loc = data[ids_comb, :]
                data_loc[:, odims[0]] += (-1)**l * L[odims[0]]
                data_loc[:, odims[1]] += (-1)**k * L[odims[1]]
                data_added.append(data_loc)
        
    for l in range(2):
        for k in range(2):
            for j in range(2):
                ids_comb = np.logical_and.reduce([ids_[l][0], ids_[k][1], ids_[j][2]])
                data_loc = data[ids_comb, :]
                data_loc[:, 0] += (-1)**l * L[0]
                data_loc[:, 1] += (-1)**k * L[1]
                data_loc[:, 2] += (-1)**j * L[2]
                data_added.append(data_loc)

    return np.vstack([data, *data_added])

def parse_args():
    parser = argparse.ArgumentParser(description="make solid mesh")
    parser.add_argument("infile", type=str, help="Input mesh")
    parser.add_argument("-Lz", default=8, type=float, help="Lz")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    num_points = 10000

    with open(args.infile, "r") as infile:
        Lx, Ly, Lz, R = [float(a) for a in infile.readline().split(" ")]
        data = []
        for line in infile.readlines():
            data.append([float(a) for a in line[:-1].split(" ")])
        data = np.array(data)

    data[:, 2] += (args.Lz - Lz)/2

    data = periodize(data, [Lx, Ly, args.Lz], R)

    #print(pos)
    points_ = []
    tets_ = []

    # From MSP - one sphere
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_points)
    theta = np.pi * (1 + 5**0.5) * indices

    points = np.empty((len(indices), 3))
    points[:, 0] = R * (np.cos(theta) * np.sin(phi))
    points[:, 1] = R * (np.sin(theta) * np.sin(phi))
    points[:, 2] = R * np.cos(phi)

    chull = ConvexHull(points)
    
    points, tris = chull.points, chull.simplices
    points = np.vstack([points, [0., 0., 0.]])
    
    tets = np.zeros((tris.shape[0], tris.shape[1]+1), dtype=int)
    print(points.shape, tris.shape, tets.shape)
    
    tets[:, :3] = tris
    tets[:, 3] = len(points)-1

    npts = 0

    for x in data:
        print(x)

        points_loc = np.copy(points[:, :])
        for dim in range(3):
            points_loc[:, dim] += x[dim]
        
        points_.append(points_loc)
        tets_.append(tets + npts)

        npts += len(points)

    points = np.vstack(points_)
    tets = np.vstack(tets_)

    if True:
        #cells = [("triangle", tris)]
        cells = [("tetra", tets)]
        meshio.write_points_cells("sphere.xdmf", points, cells)

        plt.plot(points[:, 0], points[:, 1], '.')
        plt.show()

        exit()