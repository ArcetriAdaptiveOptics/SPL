"""
Autors
  - C. Selmi:  written in 2020

Function for data smoothing::

    form SPL.ground import smooth_function
    ss = smooth_function.smooth(data, WSZ)
"""

import numpy as np



def smooth(a, WSZ):
    ''''

    Parameters
    ----------
    a: NumPy 1-D array
        containing the data to be smoothed
    WSZ: int
        smoothing window size needs, which must be odd number,
        as in the original MATLAB implementation

    Returns
    -------
    smooth: numpy array
            smoothd data
    '''
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid')/WSZ
    r = np.arange(1, WSZ-1, 2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))
