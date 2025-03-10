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

IPCAMERA = config['servers']['ip_server_camera']
PORTCAMERA = int(config['servers']['port_server_camera'])
IPFILTRO = config['servers']['ip_server_filtro']
PORTFILTRO = int(config['servers']['port_server_filtro'])
TNFRINGES = config['spl']['tn_fringes']
MEASUREMENT_ROOT_FOLDER = config['spl']['measurement_path']