import numpy as np

def calculate_mean(x, method="arithmetic"):
    if method == "arithmetic":
        return np.mean(x)
    elif method == "geometric":
        return np.prod(x)**(1/len(x))
    elif method == "harmonic":
        return len(x) / np.sum(1 / x)
    else:
        raise ValueError("Invalid method: {}".format(method))

