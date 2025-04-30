"""
Contains the main function for data acquisition.
Acquires a full-frame FITS cube by scanning wavelengths, mirroring
the hardware interaction patterns of the original SPL system.
"""

import logging
import time
import numpy as np
import os
from astropy.io import fits
import matplotlib.pyplot as plt
import sys

# Assuming plico_motor and pysilico are installed and accessible
from plico_motor import motor
import pysilico

# Import config and tracking number utility using relative paths
# This structure requires running this script as part of the spl package
# or using the sys.path modification in the __main__ block for direct execution.
from spl.conf import configuration as config
from spl.ground import tracking_number_folder


def acquire_data():
    """
    Main function to control hardware and acquire a full-frame FITS cube.

    Reads configuration from spl.conf.configuration, initializes hardware,
    optionally displays a reference frame, scans wavelengths acquiring frames,
    and saves the resulting data cube.

    Uses hardware interaction patterns from the original SPL code.

    Returns:
        str: The tracking number of the created measurement folder, or None if failed.
    """
    logging.info("Starting data acquisition process...")

    # --- Configuration (Loaded via import) --- 
    try:
        lambda_min = config.LAMBDAMIN
        lambda_max = config.LAMBDAMAX
        lambda_step = config.LAMBDASTEP
        exptime_ms = config.EXPTIME
        numframes = config.NUMFRAMES
        measurement_path = config.MEASUREMENT_ROOT_FOLDER
        show_reference = config.SHOW_REFERENCE_FRAME
        filter_settle_time = config.FILTER_SETTLE_TIME_S
        reference_lambda = config.REFERENCE_LAMBDA

        logging.info(f"Config loaded: Lambda {lambda_min}-{lambda_max}nm step {lambda_step}nm, ExpTime {exptime_ms}ms, NumFrames {numframes}")
    except AttributeError as e:
        logging.error(f"Configuration error: Missing attribute in spl.conf.configuration: {e}")
        logging.error("Please ensure required attributes (LAMBDAMIN, LAMBDAMAX, LAMBDASTEP, EXPTIME, NUMFRAMES, MEASUREMENT_ROOT_FOLDER, SHOW_REFERENCE_FRAME, FILTER_SETTLE_TIME_S, REFERENCE_LAMBDA, IPCAMERA, PORTCAMERA, IPFILTRO, PORTFILTRO) are defined in spl/conf/configuration.py")
        return None
    except Exception as e:
        logging.error(f"Error reading configuration values: {e}")
        return None

    # --- Initialize Hardware (Mirrors SPL_controller.py pattern) --- 
    camera = None
    filter_motor = None
    try:
        logging.info(f"Connecting to camera at {config.IPCAMERA}:{config.PORTCAMERA}")
        camera = pysilico.camera(config.IPCAMERA, config.PORTCAMERA)
        logging.info("Camera connected.")
        
        logging.info(f"Connecting to filter motor at {config.IPFILTRO}:{config.PORTFILTRO}")
        filter_motor = motor(config.IPFILTRO, config.PORTFILTRO, axis=1)
        logging.info("Filter motor connected.")
        
    except Exception as e:
        logging.error(f"Hardware initialization failed: {e}")
        # Optional: Add cleanup/close connections if partially successful
        return None

    # --- Optional Reference Frame --- 
    if show_reference:
        # Ensure reference lambda is int for motor
        ref_wl_int = int(round(reference_lambda)) 
        logging.info(f"Acquiring reference frame at {ref_wl_int}nm...")
        try:
            filter_motor.move_to(ref_wl_int)
            logging.info(f"Waiting {filter_settle_time}s for filter to settle at {ref_wl_int}nm...")
            time.sleep(filter_settle_time)
            camera.setExposureTime(exptime_ms) # Uses setExposureTime like original
            ref_frame = camera.getFutureFrames(1).toNumpyArray() # Uses getFutureFrames like original
            if ref_frame.ndim > 2:
                 ref_frame = np.mean(ref_frame, axis=-1) # Average if camera returns cube
                 
            logging.info("Displaying reference frame (close window to continue). Moving filter to lambda_min in background...")
            # Move filter to start position while user checks frame
            filter_motor.move_to(lambda_min) 
            
            plt.imshow(ref_frame, cmap='gray')
            plt.title(f"Reference Frame at {ref_wl_int}nm (Close to Continue)")
            plt.show() # Blocks until closed
            logging.info("Reference frame window closed.")
            
            # Wait for the move started before plt.show() to complete
            logging.info(f"Ensuring filter is at {lambda_min}nm (waiting {filter_settle_time}s max)...")
            time.sleep(filter_settle_time) 
            
        except Exception as e:
            logging.error(f"Failed during reference frame acquisition/display: {e}")
            logging.warning("Continuing acquisition without reference frame check after error.")
            # Ensure filter is moved to start if error occurred after move_to(lambda_min)
            try:
                 filter_motor.move_to(lambda_min)
                 logging.info(f"Waiting {filter_settle_time}s for filter to settle at {lambda_min}nm...")
                 time.sleep(filter_settle_time)
            except Exception as e2:
                 logging.error(f"Failed to move filter to lambda_min after reference error: {e2}")
                 return None
    else:
        # If not showing reference, ensure filter starts at lambda_min
        try:
            logging.info(f"Moving filter to starting wavelength {lambda_min}nm...")
            filter_motor.move_to(lambda_min)
            logging.info(f"Waiting {filter_settle_time}s for filter to settle...")
            time.sleep(filter_settle_time)
        except Exception as e:
            logging.error(f"Failed to move filter to starting wavelength: {e}")
            return None

    # --- Create Storage Folder (Using original pattern) --- 
    try:
        # Correctly use the config attribute directly
        fits_storage_path, tracking_number = tracking_number_folder.createFolderToStoreMeasurements(config.MEASUREMENT_ROOT_FOLDER)
        logging.info(f"Created data storage folder: {fits_storage_path} (TN: {tracking_number})")
    except Exception as e:
        logging.error(f"Failed to create storage folder: {e}")
        return None

    # --- Wavelength Scan and Acquisition --- 
    # Ensure wavelength vector uses integers
    lambda_vector = np.arange(lambda_min, lambda_max + 1, lambda_step, dtype=int)
    successful_wavelengths = []

    logging.info(f"Starting wavelength scan: {lambda_vector}")
    camera.setExposureTime(exptime_ms)
    
    for wl in lambda_vector:
        try:
            logging.info(f"Moving filter to {wl}nm...")
            filter_motor.move_to(wl)
            # Consider adding a shorter sleep if filter needs settle time between steps?
            # time.sleep(0.5) 
            
            logging.info(f"Acquiring {numframes} frame(s) at {wl}nm...")
            frames = camera.getFutureFrames(numframes).toNumpyArray()
            
            if frames.ndim == 3: 
                frame_avg = np.mean(frames, axis=2)
            elif frames.ndim == 2: 
                 frame_avg = np.mean(frames, axis=0)
            else: 
                 logging.warning(f"Unexpected frame shape {frames.shape} at {wl}nm. Skipping.")
                 continue
                 
            # Save frame immediately instead of appending to list
            # Create FITS HDU for the single frame
            hdu = fits.PrimaryHDU(frame_avg)
            # Add relevant config parameters to header
            hdu.header['WAVELEN'] = (wl, 'Wavelength (nm)')
            hdu.header['EXPTIME'] = (exptime_ms, 'Exposure time (ms)')
            hdu.header['NFRAMES'] = (numframes, 'Frames averaged per wavelength')
            hdu.header['LMIN'] = (lambda_min, 'Requested Min Lambda (nm)')
            hdu.header['LMAX'] = (lambda_max, 'Requested Max Lambda (nm)')
            hdu.header['LSTEP'] = (lambda_step, 'Requested Lambda Step (nm)')
            
            # Construct filename using integer wavelength
            output_filename = f"image_wl_{int(wl)}.fits"
            output_path = os.path.join(fits_storage_path, output_filename)
            
            # Save FITS file for this wavelength
            hdu.writeto(output_path, overwrite=True)
            logging.info(f"  Saved frame to: {output_path}")

            # Add wavelength to list *after* successful save
            successful_wavelengths.append(wl)
            # logging.info(f"Frame acquired successfully at {wl}nm.") # Logged within save message
            
        except Exception as e:
            logging.warning(f"Failed to acquire or save frame at {wl}nm: {e}. Skipping wavelength.")
            
    if not successful_wavelengths:
        logging.error("No frames were acquired successfully.")
        return None

    # --- Save Wavelength List --- 
    try:
        # Save the list of successfully acquired wavelengths (as integers)
        wavelength_list_path = os.path.join(fits_storage_path, "wavelengths.fits")
        wl_hdu = fits.PrimaryHDU(np.array(successful_wavelengths, dtype=int))
        wl_hdu.header['EXTNAME'] = 'WAVELENGTHS'
        wl_hdu.header['BUNIT'] = 'nm'
        wl_hdu.writeto(wavelength_list_path, overwrite=True)
        logging.info(f"Successfully acquired wavelengths saved to: {wavelength_list_path}")
    except Exception as e:
        logging.error(f"Failed to save wavelength list: {e}")
        # Continue anyway, but downstream processing might fail if it relies on this file

    # Remove stacking and saving of the large cube
    # --- Stack and Save Data --- 
    # try:
    #     data_cube = np.stack(acquired_frames, axis=0)
    #     ...
    # except Exception as e:
    #     logging.error(f"Failed to stack or save data cube: {e}")
    #     return None

    # --- Cleanup --- 
    # Original code didn't explicitly close connections, maintaining that pattern.
    # Add cleanup if needed based on plico_motor/pysilico behavior.

    logging.info("Data acquisition process completed successfully.")
    return tracking_number

if __name__ == '__main__':
    # Example usage for direct testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Fix for running script directly with relative imports ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..')) 
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added {project_root} to sys.path for direct script execution")
    # --- End of fix ---

    try:
        # No config path argument needed, it imports directly
        tn = acquire_data()
        if tn:
            print(f"Acquisition completed. Tracking Number: {tn}")
        else:
            print("Acquisition failed.")
    except Exception as e:
        logging.exception("An unexpected error occurred during direct execution:") 