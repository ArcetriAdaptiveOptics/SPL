'''
Authors
    - C. Selmi: written in 2021

HOW TO USE IT::

    from spl import SPL_controller as s
    camera = s.define_camera()
    filter = s.define_filter()
    tt, piston = s.SPL_measurement_and_analysis(camera, filter)
'''
import os
import numpy as np
from spl.SPL_data_acquirer import SplAcquirer
from spl.SPL_data_analyzer import SplAnalyzer
from spl.conf import configuration as config

def define_camera():
    ''' Function to use to define the camera with pysilico
    '''
    #far partire pysilico_server_2
    import pysilico
    cam = pysilico.camera(config.IPCAMERA, config.PORTCAMERA)
    return cam

def define_filter():
    ''' Function to use to define the tunable filter
    '''
    #far partire plico_motor_server_3
    from plico_motor import motor
    filter = motor(config.IPFILTRO, config.PORTFILTRO, axis=1)
    return filter

def SPL_measurement_and_analysis(camera=None, filter=None):
    '''Function for SPL data acquisition and analysis

    Parameters
    ----------
    camera: object
        camera object created with the command spl.define_camera()
    filter: object
        filter object created with the command spl.define_filter
    '''
    if camera is None:
        camera = define_camera()
    if filter is None:
        filter = define_filter()

    meas = SplAcquirer(filter, camera)
    lambda_vector = np.arange(config.LAMBDAMIN, config.LAMBDAMAX, 10)
    tt = meas.acquire(lambda_vector, config.EXPTIME, config.NUMFRAMES, mask=None)
    an = SplAnalyzer()
    piston, piston_smooth = an.analyzer(tt)
    print(piston, piston_smooth)
    return tt, piston

def get_fringe_matrix(tn, display=False):
    an = SplAnalyzer()
    mat = an.get_matrix(tn)
    if display:
        import matplotlib.pyplot as plt
        plt.imshow(mat)
        plt.colorbar()
        plt.show()
    return mat

def readMeasurement(tn):
    ''' Read images in a specific tracking number and return the cube
    args:
        tt = tracking number
    returns:
        cube = [pixels, pixels, n_frames=lambda]
        cube_normalized = creating with normalized images,
                          [pixels, pixels, n_frames=lambda]
    '''
    an = SplAnalyzer()
    return an.readMeasurement(tn)


