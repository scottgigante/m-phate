import numpy as np
import m_phate
import graphtools
from parameterized import parameterized

@parameterized(
    [(1,), (-1,)])
def test_m_phate(n_jobs):
    # create fake data
    n_time_steps = 50
    n_points = 20
    n_dim = 10
    n_pca = 5
    np.random.seed(42)
    data = np.cumsum(np.random.normal(
        0, 1, (n_time_steps, n_points, n_dim)), axis=0)

    # embedding
    m_phate_op = m_phate.M_PHATE(n_jobs=n_jobs, verbose=0, n_pca=n_pca)
    m_phate_data = m_phate_op.fit_transform(data)

    assert m_phate_data.shape[0] == n_points * n_time_steps
    assert m_phate_data.shape[1] == 2

    m_phate_op.set_params(intraslice_knn=m_phate_op.intraslice_knn)
    assert isinstance(m_phate_op.graph, graphtools.base.BaseGraph)
    m_phate_op.set_params(interslice_knn=m_phate_op.interslice_knn)
    assert isinstance(m_phate_op.graph, graphtools.base.BaseGraph)
    m_phate_op.set_params(n_svd=m_phate_op.n_svd)
    assert isinstance(m_phate_op.graph, graphtools.base.BaseGraph)

    G = m_phate_op.graph
    m_phate_op.set_params(intraslice_knn=m_phate_op.intraslice_knn+1)
    assert m_phate_op.graph is None
    m_phate_op.graph = G
    m_phate_op.set_params(interslice_knn=m_phate_op.interslice_knn+1)
    assert m_phate_op.graph is None

@parameterized(
    [(2,), (3,)])
def test_multislice_kernel(intraslice_knn):
    # create fake data
    n_time_steps = 50
    n_points = 20
    n_dim = 10
    np.random.seed(42)
    data = np.cumsum(np.random.normal(
        0, 1, (n_time_steps, n_points, n_dim)), axis=0)
    kernel = m_phate.kernel.multislice_kernel(m_phate.utils.normalize(data), 
                                              intraslice_knn=intraslice_knn,
                                              decay=None)

    nnz = 0
    # intraslice kernel
    for t in range(n_time_steps):
        subkernel = kernel[t*n_points:(t+1)*n_points][:,t*n_points:(t+1)*n_points]
        assert subkernel.sum() == n_points * (intraslice_knn+1)
        nnz += subkernel.nnz

    # interslice kernel
    for i in range(n_points):
        subkernel = kernel[i::n_points][:,i::n_points]
        assert subkernel.nnz == n_time_steps ** 2
        nnz += subkernel.nnz

    # diagonal is double counted
    nnz -= kernel.shape[0]
    # everything else should be zero
    assert nnz == kernel.nnz

    # check this passes through phate op
    m_phate_op = m_phate.M_PHATE(intraslice_knn=intraslice_knn,
                                 decay=None, verbose=0)
    m_phate_data = m_phate_op.fit_transform(data)

    # threshold
    kernel.data[kernel.data < 1e-4] = 0

    assert m_phate_data.shape[0] == n_points * n_time_steps
    assert m_phate_data.shape[1] == 2
    np.testing.assert_allclose((m_phate_op.graph.kernel - kernel).data, 0,
                               rtol=0, atol=1e-14)


def test_dm():
    # create fake data
    n_time_steps = 50
    n_points = 20
    n_dim = 10
    np.random.seed(42)
    data = np.cumsum(np.random.normal(
        0, 1, (n_time_steps, n_points, n_dim)), axis=0)
    kernel = m_phate.kernel.multislice_kernel(m_phate.utils.normalize(data))
    dm = m_phate.kernel.DM(graphtools.Graph(kernel, precomputed='affinity'))
    assert dm.shape == (n_time_steps * n_points, 2)


def test_normalize():
    # create fake data
    n_time_steps = 50
    n_points = 20
    n_dim = 10
    np.random.seed(42)
    data = np.cumsum(np.random.normal(
        0, 1, (n_time_steps, n_points, n_dim)), axis=0)
    data_norm = m_phate.utils.normalize(data)
    np.testing.assert_allclose(data_norm.mean(axis=2), 0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(data_norm.std(axis=2), 1, rtol=0, atol=1e-15)
