import numpy as np
import m_phate
from parameterized import parameterized

@parameterized(
    [(1,), (-1,)])
def test_m_phate(n_jobs):
    # create fake data
    n_time_steps = 50
    n_points = 20
    n_dim = 10
    np.random.seed(42)
    data = np.cumsum(np.random.normal(
        0, 1, (n_time_steps, n_points, n_dim)), axis=0)

    # embedding
    m_phate_op = m_phate.M_PHATE(n_jobs=n_jobs)
    m_phate_data = m_phate_op.fit_transform(data)

    assert m_phate_data.shape[0] == n_points * n_time_steps
    assert m_phate_data.shape[1] == 2
    
@parameterized(
    [(2,), (3,)])
def test_multislice_kernel(intraslice_knn)
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
    m_phate_op = m_phate.M_PHATE(intraslice_knn=intraslice_knn, decay=None)
    m_phate_data = m_phate_op.fit_transform(data)
    
    # threshold
    kernel.data[kernel.data < 1e-4] = 0

    assert m_phate_data.shape[0] == n_points * n_time_steps
    assert m_phate_data.shape[1] == 2
    assert (m_phate_op.graph.kernel - kernel).nnz == 0
