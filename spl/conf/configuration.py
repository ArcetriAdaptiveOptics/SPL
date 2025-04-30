'''
'''

import os
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'SPL.conf'))
EXPTIME = float(config['spl']['exptime'])
NUMFRAMES = int(config['spl']['numframes'])
LAMBDAMIN = int(config['spl']['lambda_min'])
LAMBDAMAX = int(config['spl']['lambda_max'])
LAMBDASTEP = int(config['spl']['lambda_step'])
N_ROWS = int(config['spl']['slice_rows'])
M_COLS = int(config['spl']['slice_cols'])
REFERENCE_LAMBDA = int(config['spl']['reference_lambda'])
FILTER_SETTLE_TIME_S = float(config['spl']['filter_settle_time_s'])
MEASUREMENT_ROOT_FOLDER = config['spl']['measurement_path']
SHOW_REFERENCE_FRAME = config.getboolean('spl', 'show_reference_frame')
TNFRINGES = config['spl']['tn_fringes']


IPCAMERA = config['servers']['ip_server_camera']
PORTCAMERA = int(config['servers']['port_server_camera'])
IPFILTRO = config['servers']['ip_server_filtro']
PORTFILTRO = int(config['servers']['port_server_filtro'])