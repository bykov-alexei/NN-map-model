import numpy as np

def default_processor(x, threshold=100):
    x = np.copy(x)
    x[x > threshold] = threshold
    x[np.isnan(x)] = x[np.logical_not(np.isnan(x))].mean()
    return x
