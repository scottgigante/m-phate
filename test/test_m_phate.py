import numpy as np
import m_phate


def test_m_phate():
    # create fake data
    n_time_steps = 100
    n_points = 50
    n_dim = 20
    np.random.seed(42)
    data = np.cumsum(np.random.normal(
        0, 1, (n_time_steps, n_points, n_dim)), axis=0)

    # embedding
    m_phate_op = m_phate.M_PHATE()
    m_phate_data = m_phate_op.fit_transform(data)

    assert m_phate_data.shape[0] == n_points * n_time_steps
    assert m_phate_data.shape[1] == 2
