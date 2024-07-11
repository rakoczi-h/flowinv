import numpy as np

def multiply_along_axis(A, B, axis):
    """
    Multiplies a multidimensional array (A) by a 1D array (B) along the desired axis (axis).
    """
    A = np.array(A)
    B = np.array(B)
    if axis >= A.ndim:
        raise ValueError("The axis is out of bounds")
    if A.shape[axis] != B.size:
        raise ValueError("'A' and 'B' must have the same length along the given axis")
    swapped_shape = A.swapaxes(0, axis).shape
    for dim_step in range(A.ndim-1):
        B = B.repeat(swapped_shape[dim_step+1], axis=0)\
             .reshape(swapped_shape[:dim_step+2])
    B = B.swapaxes(0, axis)
    return A * B

def rotate(p, origin=(0, 0), angle=0):
    """
    Rotates the cartesian (2d) coordinates of a point around another given point, the origin.
    angle is in radians
    """
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def moving_average(a, n=3):
    """
    Computes the moving average of a with a window size of n.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n

