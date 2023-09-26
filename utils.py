import itertools
import numpy as np
from iteration_utilities import unique_everseen, duplicates
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import joblib
import os

# ---------------------------------------- Data set scaling -----------------------------------------
def scale_data(data, mode, fit, name='', dataloc='', scaler='minmax'):
    """
    Normalises the data set.
    param: data: the data to be normalised. [number of samples, number of features]
           mode: 'model', 'survey' or 'models_parameterised', but can be anything. It just specifies how the scaler file is saved.
           fit: bool, if True then the data is fitted to and the scaler is saved, if False, a scaler file is searched for
           name: the name of the file where the original data is stored, again just specified how the scaler file is saved.
           scaler: 'minmax' or 'quantile', specifies what scaler to use
    output: scaled_data: the normalised data with [number of samples, number of features]
    """
    datadim = data.ndim
    if datadim == 1: # data has to be 2D before goes into scaler
        data = np.expand_dims(data, axis=0)
    if fit:
        if scaler == 'minmax':
            sc = MinMaxScaler()
        elif scaler == 'quantile':
            sc = QuantileTransformer(output_distribution='normal')
        else:
            raise NameError('Wrong scaler name given. It can be minmax or quantile.')
        scaled_data = sc.fit_transform(data)
        joblib.dump(sc, dataloc+mode+name+scaler+'_scaler.pkl')
    else:
        path_to_scaler = dataloc+mode+name+scaler+'_scaler.pkl'
        if not os.path.exists(path_to_scaler):
            raise FileNotFoundError('Scaler file does not exist.')
        sc = joblib.load(path_to_scaler)
        scaled_data = sc.transform(data)
    if datadim == 1:
        scaled_data = np.squeeze(scaled_data)
    return scaled_data

def inv_scale_data(data, mode, name='', dataloc='', scaler='minmax'):
    """
    Scales back a normalised data set.
    param: data: the data to be unscaled. [number of samples, number of features]
           mode: 'model', 'survey' or 'models_parameterised', but can be anything. It just specifies how the scaler file is saved.
           name: the name of the file where the original data is stored, again just specified how the scaler file is saved.
           scaler: 'minmax' or 'quantile', specifies what scaler to use
    output: inv_scaled_data: the unnormalised data with [number of samples, number of features]
    """
    path_to_scaler = dataloc+mode+name+scaler+'_scaler.pkl'
    if not os.path.exists(path_to_scaler):
        raise FileNotFoundError('Scaler file does not exist.')
    sc = joblib.load(path_to_scaler)
    datadim = data.ndim
    if datadim == 1:
        data = np.expand_dims(data, axis=0)
    inv_scaled_data = sc.inverse_transform(data)
    if datadim == 1:
        inv_scaled_data = np.squeeze(inv_scaled_data)
    return inv_scaled_data

# ------------------------------------- Computation tools ---------------------------------------
def multiply_along_axis(A, B, axis):
    """
    Multiplies a multidimensional array (A) by a 1D array (B) along the desired axis (axis).
    """
    A = np.array(A)
    B = np.array(B)
    # shape check
    if axis >= A.ndim:
        raise ValueError("The axis is out of bounds")
    if A.shape[axis] != B.size:
        raise ValueError("'A' and 'B' must have the same length along the given axis")
    # Expand the 'B' according to 'axis':
    # 1. Swap the given axis with axis=0 (just need the swapped 'shape' tuple here)
    swapped_shape = A.swapaxes(0, axis).shape
    # 2. Repeat:
    # loop through the number of A's dimensions, at each step:
    # a) repeat 'B':
    #    The number of repetition = the length of 'A' along the
    #    current looping step;
    #    The axis along which the values are repeated. This is always axis=0,
    #    because 'B' initially has just 1 dimension
    # b) reshape 'B':
    #    'B' is then reshaped as the shape of 'A'. But this 'shape' only
    #     contains the dimensions that have been counted by the loop
    for dim_step in range(A.ndim-1):
        B = B.repeat(swapped_shape[dim_step+1], axis=0)\
             .reshape(swapped_shape[:dim_step+2])
    # 3. Swap the axis back to ensure the returned 'B' has exactly the
    # same shape of 'A'
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
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n
