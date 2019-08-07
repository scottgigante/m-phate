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
