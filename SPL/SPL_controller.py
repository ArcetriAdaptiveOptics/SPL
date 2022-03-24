'''
Authors
    - C. Selmi: written in 2021

HOW TO USE IT::

    from SPL import SPL_controller as spl
    camera = spl.define_camera()
    filter = spl.define_filter()
    tt, piston = spl.SPL_measurement_and_analysis(camera, filter)
'''
import os
import numpy as np
from SPL_data_acquirer import SplAcquirer
from SPL_data_analyzer import SplAnalyzer

def define_camera():
    ''' Function to use to define the camera with pysilico
    '''
    #far partire pysilico_server_2
    import pysilico
    IPServer = 'localhost'
    port = 7110
    cam = pysilico.camera(IPServer, port)
    return cam

def define_filter():
    ''' Function to use to define the tunable filter
    '''
    #far partire plico_motor_server_3
    from plico_motor import motor
    filter = motor('localhost', 7300, axis=1)
    return filter

def SPL_measurement_and_analysis(camera, filter):
    '''Function for SPL data acquisition and analysis

    Parameters
    ----------
    camera: object
        camera object created with the command spl.define_camera()
    filter: object
        filter object created with the command spl.define_filter
    '''
    meas = SplAcquirer(filter, camera)
    lambda_vector = np.arange(530, 730, 10)
    tt = meas.acquire(lambda_vector, exptime=0.2, mask=None)
    print(tt)
    an = SplAnalyzer()
    piston, piston_smooth = an.analyzer(tt)
    print(piston, piston_smooth)
    return tt, piston
