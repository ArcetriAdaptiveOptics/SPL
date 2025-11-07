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
PIX2UM = float(config['spl']['pix2um'])
CROP_HEIGHT = int(config['spl']['crop_height'])
CROP_WIDTH = int(config['spl']['crop_width'])
DESIRED_FLUX_LEVEL = int(config['spl']['desired_flux_level'])
FILTER_BANDWIDTH_MODE = int(config['spl']['filter_bandwidth_mode'])
ENABLE_SPL_LOGGING = config.getboolean('spl', 'enable_spl_logging', fallback=True)

# New configuration parameters
POSITIONS_TO_ROTATION_ANGLES = config.get('spl', 'positions_to_rotation_angles', fallback=None)
# New parameter for vertical alignment
POSITIONS_TO_ROTATION_ANGLES_VERTICAL = config.get('spl', 'positions_to_rotation_angles_vertical', fallback=None)
#FRINGES_HEIGHT = int(config['spl']['fringes_height'])
DARK_FILENAME = config.get('spl', 'dark_filename', fallback=None)
SPOT_FIND_THRESHOLD = config.get('spl', 'spot_find_threshold', fallback=None)

IPCAMERA = config['servers']['ip_server_camera']
PORTCAMERA = int(config['servers']['port_server_camera'])
IPFILTRO = config['servers']['ip_server_filtro']
PORTFILTRO = int(config['servers']['port_server_filtro'])

# Flux calibration file
_flux_calib_file_path_str = config.get('spl', 'flux_calibration_filename', fallback=None)
FLUX_CALIBRATION_FILENAME = None
if _flux_calib_file_path_str:
    conf_dir = os.path.dirname(__file__) # Path to SPL/spl/conf/
    spl_spl_dir = os.path.dirname(conf_dir) # Path to SPL/spl/

    if os.path.isabs(_flux_calib_file_path_str):
        FLUX_CALIBRATION_FILENAME = _flux_calib_file_path_str
    else:
        # If SPL.conf has '.\calib\file.fits', we want it relative to SPL/spl/
        # So, join with spl_spl_dir instead of conf_dir
        # e.g., os.path.join(SPL/spl/, '.\calib\file.fits') -> SPL/spl/calib/file.fits
        FLUX_CALIBRATION_FILENAME = os.path.abspath(os.path.join(spl_spl_dir, _flux_calib_file_path_str))
    
    # Optional check, can be enabled for debugging path resolution
    # if not os.path.isfile(FLUX_CALIBRATION_FILENAME):
    #     print(f"WARNING: Flux calibration file specified but not found at resolved path: {FLUX_CALIBRATION_FILENAME}")
    # else:
    #     print(f"INFO: Flux calibration file path resolved to: {FLUX_CALIBRATION_FILENAME}")
else:
    print("INFO: No flux_calibration_filename specified in SPL.conf.")