import numpy as np
from matplotlib.path import Path

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

def points_within_area(xyboundary, points):
    """
    Given a boundary, and a set of points, checks which points are within the area.
    """
    path = Path(xyboundary)
    inside = path.contains_points(points)
    return inside

def distance_to_line_segment(p, a, b):
    """Cartesian distance from point to line segment

    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892

    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2), one of the end points
        - b: np.array of shape (x, 2), the other end point
    """
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))
    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)
    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])
    # perpendicular distance component
    # rowwise cross products of 2D vectors  
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]
    return np.hypot(h, c)

def normalize(x, range=(0, 1), minx=None, maxx=None):
    if minx is None:
        minx = np.min(x)
    if maxx is None:
        maxx = np.max(x)

    if minx == maxx:
        y = np.full_like(x, np.mean(range), dtype=float)
        scale = 0
        offset = np.mean(range)
    else:
        scale = (range[1] - range[0]) / (maxx - minx)
        offset = range[0] - minx * scale
        y = scale * x + offset
    return y, scale, offset

def pad_grid(grid, pad_width, square=False):
    if isinstance(pad_width, int) or isinstance(pad_width, float):
        pad_width = [int(pad_width)]
    if len(pad_width) == 1:
        pad_width = [pad_width[0], pad_width[0], pad_width[0]]

    n_x = np.shape(grid)[0]
    dx = (np.max(grid[:,:,0])-np.min(grid[:,:,0]))/(n_x-1)
    X = np.linspace(np.min(grid[:,:,0])-dx*pad_width[0], np.max(grid[:,:,0])+dx*pad_width[0], num=n_x+2*pad_width[0])
    if square:
        Y = X
    else:
        n_y = np.shape(grid)[1]
        dy = (np.max(grid[:,:,1])-np.min(grid[:,:,1]))/(n_y-1)
        Y = np.linspace(np.min(grid[:,:,1])-dy*pad_width[1], np.max(grid[:,:,1])+dy*pad_width[1], num=n_y+2*pad_width[1])
    X, Y = np.meshgrid(X, Y , indexing='ij')
    X = np.expand_dims(X, axis=2)
    Y = np.expand_dims(Y, axis=2)
    Z = np.zeros(np.shape(X))
    grid = np.c_[X, Y, Z]
    return grid
