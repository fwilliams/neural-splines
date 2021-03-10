import pykeops.torch as keops
import torch


def kmeans(x, k, num_iters=10):
    """
    Implements Lloyd's algorithm for the Euclidean metric.
    :param x: A tensor representing a set of N points of dimension D (shape [N, D])
    :param k: The number of centroids to compute
    :param num_iters: The number of K means iterations to do
    :return: cl, c where cl are cluster labels for each input point (shape [N]) and c are the
             cluster centroids (shape [K, D])
    """

    N, D = x.shape  # Number of samples, dimension of the ambient space

    # Simplistic initialization for the centroids
    perm = torch.randperm(N)[:k]
    c = x[perm, :].clone()
    cl = None

    x_i = keops.LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = keops.LazyTensor(c.view(1, k, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(num_iters):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=k).type_as(c).view(k, 1)
        print(Ncl.min(), Ncl.max())
        c /= Ncl  # in-place division to compute the average

    return cl, c
