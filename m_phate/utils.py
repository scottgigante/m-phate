import numpy as np


def normalize(X):
    X = X - np.mean(X, axis=2)[:, :, None]
    std = np.std(X, axis=2)[:, :, None]
    X = X / np.where(std > 0, std, 1)
    return X
