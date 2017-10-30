import numpy as np


def error(y_real, y_pred):
    # mean squared error
    return np.sum((y_pred - y_real) ** 2) / len(y_real)