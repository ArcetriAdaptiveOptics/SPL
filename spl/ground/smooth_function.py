"""
Autors
  - C. Selmi:  written in 2020

Function for data smoothing::

    form SPL.ground import smooth_function
    ss = smooth_function.smooth(data, WSZ)
"""

import numpy as np



def smooth(a, WSZ):
    ''''Performs moving average smoothing on a 1-D array.

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
            smoothed data, same size as input array 'a'
    '''
    # Check if window size is odd, as per original docstring
    if WSZ % 2 == 0:
        print(f"Warning: Smoothing window size WSZ={WSZ} should ideally be odd.")
        # Proceeding anyway, but np.convolve handles even sizes.
        
    # Use convolution with mode='same' to ensure output size matches input size.
    # The kernel is an array of ones divided by the window size for averaging.
    kernel = np.ones(WSZ) / WSZ
    smoothed_a = np.convolve(a, kernel, mode='same')
    
    return smoothed_a
