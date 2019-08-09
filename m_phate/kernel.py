import graphtools
import graphtools.utils
from scipy.spatial import distance
from scipy import sparse
import numpy as np
from joblib import Parallel, delayed


def graph_kernel(*args, **kwargs):
    return graphtools.Graph(*args, **kwargs).K


def square_pdist(X):
    return distance.squareform(distance.pdist(X))


def knn_dist(D, interslice_knn, fun=np.median):
    return np.median(np.partition(
        D, interslice_knn, axis=1)[:, interslice_knn])


def distance_to_kernel(D, bandwidth):
    # gaussian kernel
    A = np.exp(-1 * (D / bandwidth)**2)
    A = (A + A.T) / 2
    return A


def _diagonalize_interslice_kernels(interslice_kernels, method='dia'):
    n = interslice_kernels[0].shape[0]
    m = len(interslice_kernels)
    if method == 'dia':
        diags = np.zeros((2*n-1, n*m))
        for vertex, A in enumerate(interslice_kernels):
            diags[(np.tile(np.arange(n, 2*n), n) - np.repeat(np.arange(1, n+1), n), 
                   np.tile(np.arange(0, n*m, m), n) + vertex)] = A.flatten()
        offsets = np.arange(-n*m + m, n*m, m)
        # set main diagonal to 0
        diags[offsets == 0] = 0
        K = sparse.dia_matrix((diags, offsets), shape=(n*m, n*m))
    else:
        assert method == 'csr'
        # this is inefficient
        K = sparse.csr_matrix((n*m, n*m))
        for vertex, A in enumerate(interslice_kernels):
            K = graphtools.utils.set_submatrix(
                K, np.arange(n) * m + vertex, np.arange(n) * m + vertex, A)
        # set main diagonal to 0
        K = graphtools.utils.set_diagonal(K, 0)
    return K
        


def _multislice_kernel(data,
                       intraslice_knn=2,
                       interslice_knn=25,
                       decay=5,
                       n_pca=100,
                       p=list,
                       kernel_fn=graph_kernel,
                       pdist_fn=square_pdist,
                       knn_fn=knn_dist,
                       kernel_dist_fn=distance_to_kernel,
                       **kwargs):
    n = data.shape[0]
    m = data.shape[1]
    N = n * m
    K = sparse.lil_matrix((N, N))

    kernels = p(kernel_fn(
        x, knn=intraslice_knn, decay=decay, **kwargs, n_jobs=1)
        for x in data)
    for i, G_K in enumerate(kernels):
        # plug into K
        K = graphtools.utils.set_submatrix(K, np.arange(
            i * m, (i + 1) * m), np.arange(i * m, (i + 1) * m), G_K)

    # set interslice fixed bandwidth
    interslice_dist = p(pdist_fn(data[:, vertex, :])
                        for vertex in range(data.shape[1]))
    bandwidths = p(knn_fn(D, interslice_knn)
                   for D in interslice_dist)
    bandwidth = np.mean(bandwidths)
    interslice_kernels = p(kernel_dist_fn(
        D, bandwidth)
        for D in interslice_dist)

    K = K.tocsr() + _diagonalize_interslice_kernels(interslice_kernels).tocsr()
        
    return K


def multislice_kernel(data,
                      intraslice_knn=2,
                      interslice_knn=25,
                      decay=5,
                      n_pca=100,
                      n_jobs=20, **kwargs):
    n = data.shape[0]
    m = data.shape[1]

    if n_pca is not None and n_pca < data.shape[2]:
        data = data.reshape(n * m, -1)
        data = graphtools.base.Data(data, n_pca=n_pca).data_nu
        data = data.reshape(n, m, n_pca)

    # build within slice graphs
    if n_jobs == 1:
        K = _multislice_kernel(data,
                               intraslice_knn=intraslice_knn,
                               interslice_knn=interslice_knn,
                               decay=decay,
                               n_pca=n_pca)
    else:
        with Parallel(n_jobs=n_jobs) as p:
            K = _multislice_kernel(
                data,
                intraslice_knn=intraslice_knn,
                interslice_knn=interslice_knn,
                decay=decay,
                n_pca=n_pca,
                p=p,
                kernel_fn=delayed(graph_kernel),
                pdist_fn=delayed(square_pdist),
                knn_fn=delayed(knn_dist),
                kernel_dist_fn=delayed(distance_to_kernel))
    return K.tocsr()


def DM(G, t=1, n_components=2):
    # symmetric affinity matrix
    diff_aff = sparse.csr_matrix(G.diff_aff)
    # symmetrize to remove numerical error
    diff_aff = (diff_aff + diff_aff.T) / 2
    # svd
    U, S, _ = sparse.linalg.svds(diff_aff, k=n_components + 1)
    # order in terms of smallest eigenvalue
    U, S = U[:, ::-1], S[::-1]
    # get first eigenvector
    u1 = U[:, 0][:, None]
    # ensure non-zero
    zero_idx = np.abs(u1) <= np.finfo(float).eps
    u1[zero_idx] = (np.sign(u1[zero_idx]) * np.finfo(
        float).eps).reshape(-1)
    # normalize by first eigenvector
    U = U / u1
    # drop first eigenvector
    U, S = U[:, 1:], S[1:]
    # power eigenvalues
    S = np.power(S, t)
    # weight U by eigenvalues
    dm = U.dot(np.diagflat(S))
    return dm
