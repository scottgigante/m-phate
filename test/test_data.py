import m_phate.data
from parameterized import parameterized

@parameterized(
    [(m_phate.data.load_mnist,), 
     (m_phate.data.load_cifar,)])
def test_data(data_fn):
    x_train, x_test, y_train, y_test = data_fn()
    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]
    assert x_train.shape[1] == x_test.shape[1]
    assert y_train.shape[1] == y_test.shape[1]
    assert x_train.shape[0] > x_test.shape[0]
