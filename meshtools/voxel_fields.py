import numpy as np
from .voxels import laplacian_filter, get_clusters, get_connected_clusters


def smoothed_crossing_pipes(X, Y, Z, R_left, R_right):
    R_Z = np.sqrt((X-0.5)**2 + (Y-0.5)**2)
    R_Y = np.sqrt((X-0.5)**2 + (Z-0.5)**2)
    R_X = np.sqrt((Y-0.5)**2 + (Z-0.5)**2)

    R_max_Z = (R_left-R_right)*(1+np.cos(np.pi*Z))/2 + R_right
    R_max_Y = (R_left-R_right)*(1+np.cos(np.pi*Y))/2 + R_right
    R_max_X = (R_left-R_right)*(1+np.cos(np.pi*X))/2 + R_right

    S_Z = - R_Z + R_max_Z > 0
    S_Y = - R_Y + R_max_Y > 0
    S_X = - R_X + R_max_X > 0
    S = -(1-S_X)*(1-S_Y)*(1-S_Z) + 0.5

    for i in range(40):
        S = laplacian_filter(S, 0.05)

    return S


def percolation_cluster(N, dim, p, axis=0):
    size = (N,)*dim
    if N == 1:
        return np.ones(size, dtype=bool)

    conn = False
    while not conn:
        R = np.random.rand(*size)
        bw = R < p

        clusters, cluster_conn = get_clusters(bw, axis=axis)
        conn = np.any(cluster_conn)

    S = get_connected_clusters(clusters, cluster_conn)
    return S
