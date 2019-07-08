import scprep
scprep.utils.check_version("keras", "2.2")  # noqa

import keras


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1) / 255
    x_test = x_test.reshape(x_test.shape[0], -1) / 255
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    return x_train, x_test, y_train, y_test
