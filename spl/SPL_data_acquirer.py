'''
Authors
  - G. Pariani, R.Briguglio: written in 2016
  - C. Selmi: ported to Python in 2020
  - A. Puglisi and C. Selmi: python code debugging in 2021
'''

import os
import logging
import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import scipy
import scipy.ndimage as scin
from astropy.io import fits as pyfits
from photutils import centroids
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm
from spl.ground import tracking_number_folder
from spl.conf import configuration as config


class SplAcquirer():
    '''
    Class used to acquire images, whit SPL camera, at different wavelengths.

    HOW TO USE IT::
        from SPL.SPL_data_acquirer import SplAcquirer
        #definizione dell'oggetto filtro e dell'oggetto camera
        spl = SplAcquirer(filtro, camera)
        lambda_vector = np.arange(530, 730, 10)
        tt = spl.acquire(lambda_vector, exptime=0.7, mask=None)

    '''

    def __init__(self, filter_obj, camera_obj):
        """The constructor """
        self._logger = logging.getLogger('SPL_ACQ:')
        self._filter = filter_obj
        self._camera = camera_obj
        self._pix2um = config.PIX2UM  # Using config value
        self._exptime = None
        self._n_rows = config.N_ROWS
        self._m_cols = config.M_COLS
        self._flux_calibration_data = self._load_flux_calibration()
        self._CAMERA_EXPTIME_MIN_MS = 0.001  # Minimum exposure time in ms (1 microsecond)
        self._CAMERA_EXPTIME_MAX_MS = 1000.0 # Maximum exposure time in ms (1,000,000 microseconds or 1 second)

    def _load_flux_calibration(self):
        """Loads flux calibration data from the FITS file specified in config."""
        calib_filepath = config.FLUX_CALIBRATION_FILENAME
        if not calib_filepath:
            self._logger.info("Flux calibration file not specified in configuration. Proceeding without flux calibration.")
            return None

        if not os.path.isfile(calib_filepath):
            self._logger.warning(f"Flux calibration file not found at '{calib_filepath}'. Proceeding without flux calibration.")
            return None

        try:
            with pyfits.open(calib_filepath) as hdul:
                if len(hdul) < 2 or 'MEAN_WAVELENGTH_CORRECTIONS' not in hdul:
                    self._logger.error(f"Flux calibration FITS file '{calib_filepath}' does not contain the required HDU 'MEAN_WAVELENGTH_CORRECTIONS'.")
                    return None
                
                calib_table_hdu = hdul['MEAN_WAVELENGTH_CORRECTIONS']
                data = calib_table_hdu.data
                
                # Check for required columns
                required_cols = ['WAVELENGTH_NM', 'MEAN_EXPOSURE_CORR_FACTOR']
                missing_cols = [col for col in required_cols if col not in data.columns.names]
                if missing_cols:
                    self._logger.error(f"Flux calibration table in '{calib_filepath}' is missing required columns: {missing_cols}.")
                    return None

                calibration_map = {row['WAVELENGTH_NM']: row['MEAN_EXPOSURE_CORR_FACTOR'] 
                                   for row in data}
                self._logger.info(f"Successfully loaded flux calibration data from '{calib_filepath}' for {len(calibration_map)} wavelengths.")
                return calibration_map
        except Exception as e:
            self._logger.error(f"Error loading flux calibration file '{calib_filepath}': {e}")
            return None

    @staticmethod
    def _storageFolder(path=None):
        """ Creates the path where to save measurement data"""
        measurement_path = config.MEASUREMENT_ROOT_FOLDER
        return measurement_path
     
    # def _populate_fits_header_from_snapshot(self, header, snapshot_dict, current_prefix):
    #     """
    #     Recursively populates a FITS header with key-value pairs from a snapshot dictionary,
    #     using HIERARCH FITS convention for clarity.

    #     Args:
    #         header (pyfits.Header): The FITS header object to populate.
    #         snapshot_dict (dict): The dictionary containing snapshot data.
    #         current_prefix (str): The prefix string for the current level of keys (e.g., 'CAMSET', 'CAMSET STATUS').
    #     """
    #     if not isinstance(snapshot_dict, dict):
    #         # self._logger.debug(f"Snapshot part for prefix '{current_prefix}' is not a dict, skipping: {snapshot_dict}")
    #         return

    #     for key, value in snapshot_dict.items():
    #         # Construct a descriptive FITS keyword string using the prefix and current key
    #         # Replace problematic characters for FITS keywords but aim for readability
    #         # Spaces are okay as astropy will handle HIERARCH
    #         clean_key_part = str(key).strip()
    #         # Basic cleaning: replace multiple spaces with one, remove leading/trailing problematic chars for key parts
    #         clean_key_part = re.sub(r'\s+', ' ', clean_key_part)
    #         clean_key_part = re.sub(r'[^a-zA-Z0-9_.: -]', '', clean_key_part) # Allow a few more chars for HIERARCH

    #         if not clean_key_part: # If key becomes empty after cleaning
    #             # self._logger.warning(f"Skipping empty key generated from original key '{key}' under prefix '{current_prefix}'.")
    #             continue

    #         # Form the full keyword path, astropy handles HIERARCH for long ones / with spaces
    #         fits_keyword_str = f"{current_prefix.upper()} {clean_key_part.upper()}"
    #         # Limit total HIERARCH keyword string length (conventionally up to ~68-70 chars for keyword part)
    #         fits_keyword_str = fits_keyword_str[:68].strip()

    #         comment_str = f"Snapshot value for {current_prefix} {key}"[:68]

    #         if isinstance(value, dict):
    #             # If the value is another dictionary, recurse
    #             self._populate_fits_header_from_snapshot(header, value, fits_keyword_str)
    #         else:
    #             # Process primitive values
    #             processed_value = None
    #             value_comment_suffix = ""

    #             if value is None:
    #                 processed_value = 'N/A' # FITS doesn't have None, use a placeholder string
    #                 value_comment_suffix = " (was None)"
    #             elif isinstance(value, bool):
    #                 processed_value = value # astropy handles T/F
    #             elif isinstance(value, (int, float, np.number)):
    #                 if np.isnan(value) or np.isinf(value):
    #                     processed_value = 'N/A'
    #                     value_comment_suffix = " (was NaN/Inf)"
    #                 else:
    #                     processed_value = value
    #             elif isinstance(value, str):
    #                 # Ensure ASCII compliance and truncate
    #                 # Make control characters visible and remove them, then encode to ASCII
    #                 s = repr(value)[1:-1] # Get string content without quotes, e.g. '\r\n'
    #                 s = s.replace('\\r', '').replace('\\n', '').replace('\\t', '') # Remove 
 
    #                 # Replace other common non-printable or problematic characters if necessary
    #                 # For now, focus on what repr and encode handle.
    #                 ascii_value = s.encode('ascii', 'replace').decode('ascii')
    #                 processed_value = ascii_value[:68] # FITS string values also have length limits
    #             else:
    #                 try:
    #                     str_val = str(value)
    #                     ascii_value = str_val.encode('ascii', 'replace').decode('ascii')
    #                     processed_value = ascii_value[:68]
    #                     value_comment_suffix = f" (original type: {type(value).__name__})"
    #                 except Exception as e_conv:
    #                     self._logger.warning(f"Could not convert value for '{fits_keyword_str}' to string: {e_conv}. Skipping.")
    #                     continue # Skip this key-value pair
                
    #             try:
    #                 # Check if key already exists (e.g. from a previous, less specific add)
    #                 # This simple check might not be sufficient for complex HIERARCH overlaps.
    #                 if fits_keyword_str in header:
    #                     # self._logger.debug(f"FITS keyword '{fits_keyword_str}' already exists. Overwriting.")
    #                     pass # Allow overwrite, astropy handles it
    #                 header[fits_keyword_str] = (processed_value, comment_str + value_comment_suffix)
    #             except Exception as e:
    #                 self._logger.warning(f"Could not add FITS entry for keyword '{fits_keyword_str}' (Original key: '{key}'): {e} (Value: {value}, Type: {type(value)})")

    def _set_camera_exposure_time(self, exptime_ms: float) -> float:
        """
        Helper method to set camera exposure time with proper limits.
        
        Args:
            exptime_ms (float): Desired exposure time in milliseconds
            
        Returns:
            float: The actual exposure time that was set (may be limited)
        """
        # Ensure exposure time is within camera limits
        if exptime_ms < self._CAMERA_EXPTIME_MIN_MS:
            self._logger.info(f"Requested exposure time {exptime_ms:.3f}ms is below minimum ({self._CAMERA_EXPTIME_MIN_MS:.3f}ms). Using minimum.")
            exptime_ms = self._CAMERA_EXPTIME_MIN_MS
        elif exptime_ms > self._CAMERA_EXPTIME_MAX_MS:
            self._logger.info(f"Requested exposure time {exptime_ms:.3f}ms exceeds maximum ({self._CAMERA_EXPTIME_MAX_MS:.1f}ms). Using maximum.")
            exptime_ms = self._CAMERA_EXPTIME_MAX_MS
            
        try:
            self._camera.setExposureTime(exptime_ms)
            # self._logger.debug(f"Successfully set camera exposure time to {exptime_ms:.3f}ms.")
            return exptime_ms
        except Exception as e:
            # Log the value of exptime_ms that caused the error *before* fallback.
            self._logger.error(f"Failed to set exposure time to {exptime_ms:.3f}ms (intended value): {e}")
            # Fallback calculation logic:
            # This re-applies clipping, which is fine. If exptime_ms was already clipped, safe_exptime will be the same.
            safe_exptime = min(max(exptime_ms, self._CAMERA_EXPTIME_MIN_MS), self._CAMERA_EXPTIME_MAX_MS)
            self._logger.info(f"Attempting to fall back to safe exposure time: {safe_exptime:.3f}ms")
            try:
                self._camera.setExposureTime(safe_exptime)
                self._logger.info(f"Successfully set fallback exposure time to {safe_exptime:.3f}ms.")
                return safe_exptime
            except Exception as e_fallback:
                self._logger.critical(f"CRITICAL: Failed to set fallback exposure time {safe_exptime:.3f}ms: {e_fallback}. Camera may be unresponsive.")
                # Depending on desired behavior, could raise an error or return a "known bad" value
                # For now, return the problematic 'safe_exptime' as it's the last attempted.
                return safe_exptime 

    def acquire(self, lambda_vector=None, exptime=None, numframes=None, mask=None, display_reference=False,
                actuator_position=None): # Changed from actuator_snapshot to actuator_position
        """
        Acquires images at different wavelengths, subtracts dark frame, 
        and saves them as FITS cubes.
        
        Args:
            lambda_vector: Array of wavelengths to acquire. If None, uses config values
            exptime: Exposure time in milliseconds. If None, uses config value
            numframes: Number of frames to average. If None, uses config value
            mask: Optional mask to apply
            display_reference: If True, displays the reference image after dark subtraction
            actuator_position (float, optional): Position of the actuator in nanometers.
        """
        # Use configuration values if parameters are not provided
        if lambda_vector is None:
            lambda_vector = np.arange(config.LAMBDAMIN, config.LAMBDAMAX + 1, config.LAMBDASTEP)
        if exptime is None:
            exptime = config.EXPTIME
        if numframes is None:
            numframes = config.NUMFRAMES

        # Now subtract the dark frame from subsequent frames during acquisition
        self._set_camera_exposure_time(exptime)
        dark_frame = self._capture_dark_frame(exptime)
        self._dark_frame = dark_frame  # Store the dark frame for later use

        # Create a new folder for the measurement
        self._dove, tt = tracking_number_folder.createFolderToStoreMeasurements(self._storageFolder())
        fits_file_name = os.path.join(self._dove, 'lambda_vector.fits')
        pyfits.writeto(fits_file_name, lambda_vector)

        # Step 1: Capture reference frame
        reference_image = self._captureReferenceFrame(config.REFERENCE_LAMBDA, exptime)
        
        # Ensure reference image is float32 and 2D
        if reference_image.ndim == 3:
            reference_image = np.mean(reference_image, axis=2) # Assuming (H, W, N_frames)
        reference_image = reference_image.astype(np.float32)
        
        # Subtract the dark frame from the reference image and clip negative values to zero
        reference_image -= self._dark_frame
        reference_image = np.clip(reference_image, 0, None)
        
        # Step 1.5: Slice the reference image and find spots in each subframe
        subframe_positions = self._sliceAndFindSpotPositions(reference_image, n=self._n_rows, m=self._m_cols)  
        self._logger.info(f"Looking for spots in {len(subframe_positions)} subframes.")
        
        # Check if any spots were found in any subframe
        if not any(subframe_positions):
             self._logger.error("No bright spots detected in ANY subframe!")
             raise ValueError("No bright spots detected in ANY subframe!")
        
        # Step 2: Combine spot positions from subframes and convert to full frame coordinates
        positions = []
        subframe_height = reference_image.shape[0] // self._n_rows
        subframe_width = reference_image.shape[1] // self._m_cols
        
        for k, spots_in_subframe in enumerate(subframe_positions):
            if not spots_in_subframe:
                continue 
            if self._m_cols == 0:
                self._logger.error("self._m_cols is zero, cannot calculate subframe column index. Skipping subframe processing.")
                continue
            row_idx = k // self._m_cols
            col_idx = k % self._m_cols
            for (cx_subframe, cy_subframe) in spots_in_subframe:
                full_frame_y = row_idx * subframe_height + cy_subframe 
                full_frame_x = col_idx * subframe_width + cx_subframe
                positions.append((full_frame_x, full_frame_y))
        
        if not positions:
            self._logger.error("No valid spot coordinates generated after processing subframes!")
            raise ValueError("No valid spot coordinates generated after processing subframes!")
            
        self._logger.info(f"Total spot positions identified: {len(positions)}")

        if display_reference or config.SHOW_REFERENCE_FRAME:
            self._display_reference_image(reference_image, positions, self._n_rows, self._m_cols)

        # Step 3: Scan through wavelengths and preprocess data
        frame_cube = []
        for wl_idx, wl in enumerate(tqdm(lambda_vector, desc="Acquiring Wavelengths", unit="wvln")):
            self._logger.info(f"Acquiring image at {wl} nm...")
            
            current_exptime = exptime # Base exptime for this wavelength

            # Adjust exposure time based on flux calibration data, if available
            if self._flux_calibration_data:
                correction_factor = self._flux_calibration_data.get(wl, 1.0) # Default to 1.0 if wl not in calib
                if correction_factor != 1.0:
                    adjusted_exptime = int(exptime * correction_factor) # exptime is the base exptime
                    self._logger.info(f"Flux calib: wl={wl}nm, factor={correction_factor:.2f}, base_exp={exptime}ms, adjusted_exp={adjusted_exptime}ms")
                    if adjusted_exptime > 0: # Ensure exposure time is positive
                        current_exptime = adjusted_exptime
                        # Max limit check removed here, _set_camera_exposure_time will handle it.
                    else:
                        self._logger.warning(f"Flux calibration for {wl}nm resulted in non-positive exptime ({adjusted_exptime}ms). Using base exptime {exptime}ms for this iteration.")
                        # current_exptime remains the initial 'exptime' for this iteration
                else:
                    self._logger.debug(f"No flux calibration correction for {wl}nm or factor is 1.0. Using base exptime {exptime}ms.")
            
            # Set filter and acquire image
            filter_move_start_time = time.time()
            self._filter.move_to(wl)
            filter_move_end_time = time.time()
            self._logger.info(f"TIMING: self._filter.move_to({wl}) took {filter_move_end_time - filter_move_start_time:.4f} s")

            # Set exposure time - this might be redundant if flux calib did it,
            # but ensures it's set if no calib or if calib resulted in no change / invalid value.
            exposure_set_start_time = time.time()
            actual_exptime = self._set_camera_exposure_time(current_exptime)
            exposure_set_end_time = time.time()
            self._logger.info(f"TIMING: self._camera.setExposureTime({actual_exptime}) took {exposure_set_end_time - exposure_set_start_time:.4f} s")
            
            frame_acq_start_time = time.time()
            image = self._camera.getFutureFrames(numframes).toNumpyArray()
            frame_acq_end_time = time.time()
            self._logger.info(f"TIMING: self._camera.getFutureFrames({numframes}) took {frame_acq_end_time - frame_acq_start_time:.4f} s (Exposure: {actual_exptime}ms)")

            if image.ndim == 3:
                 image = np.mean(image, axis=2) # Assuming (H, W, N_frames)
            image = image.astype(np.float32) # Ensure image is float32 before dark subtraction

            image -= self._dark_frame
            image = np.clip(image, 0, None)

            processed_wavelength_frames = []
            for idx, pos in enumerate(positions):
                cx, cy = pos
                crop = self._preProcessing(image, cy, cx)
                file_name = 'image_%dnm_pos%02d.fits' % (wl, idx)
                self._saveCameraFrame(file_name, crop, wavelength=wl, exptime_ms=actual_exptime,
                                      actuator_position=actuator_position) # Pass actuator_position
                processed_wavelength_frames.append(crop)
            
            frame_cube.append(processed_wavelength_frames)

        self._logger.info(f"Data saved in {self._dove}")
        self._logger.info("Acquisition complete.")
        return tt

    def _display_reference_image(self, reference_image, spot_positions, n_rows_slice, n_cols_slice):
        """Displays the reference image with slice boundaries and spot IDs."""
        try:
            plt.figure(figsize=(10, 8))
            plt.imshow(reference_image, cmap='gray', origin='lower')
            plt.colorbar(label='Intensity')
            
            # --- Add Slice Boundary Visualization ---
            height, width = reference_image.shape
            row_size = height // n_rows_slice
            col_size = width // n_cols_slice

            # Draw horizontal lines
            for i in range(1, n_rows_slice):
                plt.axhline(y=i * row_size, color='r', linestyle='--', alpha=0.7)
            # Draw vertical lines
            for j in range(1, n_cols_slice):
                plt.axvline(x=j * col_size, color='r', linestyle='--', alpha=0.7)
            # --- End Slice Boundary Visualization ---

            # --- Add Spot Position IDs ---
            for idx, (x, y) in enumerate(spot_positions):
                plt.text(x, y + 5, str(idx), color='yellow', fontsize=8, ha='center', va='bottom',
                         bbox=dict(facecolor='black', alpha=0.3, pad=0.1, edgecolor='none'))
            # --- End Spot Position IDs ---

            plt.title(f'Reference Image (Dark Subtracted) with {n_rows_slice}x{n_cols_slice} Slice Boundaries & Spot IDs')
            plt.xlabel("X pixel")
            plt.ylabel("Y pixel")
            plt.show()
        except Exception as e:
            self._logger.warning(f"Could not display reference image with slice boundaries and spot IDs: {e}")

    def _sliceAndFindSpotPositions(self, image, n, m):
        """
        Slices the reference image into n x m subframes and finds spots in each subframe.
        
        Args:
            image: The reference image to slice and process.
            n: Number of rows for each subframe.
            m: Number of columns for each subframe.
            
        Returns:
            A list of lists, where each inner list contains the spot positions found in a subframe.
        """
        subframe_positions = []
        height, width = image.shape

        # Calculate subframe size
        row_size = height // n
        col_size = width // m

        # Loop over each subframe and find spots
        for i in range(n):
            for j in range(m):
                # Define subframe boundaries
                row_start = i * row_size
                row_end = (i + 1) * row_size if i < n - 1 else height
                col_start = j * col_size
                col_end = (j + 1) * col_size if j < m - 1 else width
                
                # Slice the image
                subframe = image[row_start:row_end, col_start:col_end]
                
                # Find spots in the subframe
                spots = self._findSpotPositions(subframe)
                subframe_positions.append(spots)
        
        return subframe_positions

    def _captureReferenceFrame(self, wavelength, exptime):
        """Captures a reference image at a given wavelength."""
        self._logger.info(f'Acquiring reference image at {wavelength} nm...')
        self._logger.info(f'Exposure time: {exptime} ms')
        self._filter.move_to(wavelength)
        time.sleep(config.FILTER_SETTLE_TIME_S)  # Using config value
        self._set_camera_exposure_time(exptime)
        image = self._camera.getFutureFrames(1).toNumpyArray()
        return image

    def _findSpotPositions(self, image):
        """Finds the positions of bright spots in a given image."""
        #print("Finding bright spots...")
        #plt.imshow(image, cmap='gray')
        #plt.title(f'Reference subframe')
        #plt.show()       
        return self._findBrightSpots(image, threshold=100, npixels=500)

    def _saveFitsCubes(self, lambda_vector, frame_cube):
        """Saves the acquired frames as a FITS file."""
        fits_file_name = os.path.join(self._dove, 'acquired_data.fits')
        pyfits.writeto(fits_file_name, frame_cube, overwrite=True)
        pyfits.append(fits_file_name, lambda_vector)

    def _saveCameraFrame(self, file_name, frame, wavelength=None, exptime_ms=None,
                         actuator_position=None): # Changed parameters
        ''' Save camera frames in SPL/TrackingNumber
        '''
        fits_file_name = os.path.join(self._dove, file_name)
        if isinstance(frame, np.ma.MaskedArray):
            data = frame.data
            mask = frame.mask.astype(np.int32)
        else:
            data = frame
            mask = None

        header = pyfits.Header()
        if wavelength is not None:
            header['WAVELEN'] = (wavelength, 'Wavelength in nanometers')
        if exptime_ms is not None:
            header['EXPTIME'] = (exptime_ms, 'Exposure time in milliseconds')
        if actuator_position is not None:
            header['ACTPOS'] = (actuator_position, 'Actuator position in nanometers')

        pyfits.writeto(fits_file_name, data, header=header, overwrite=True)
        if mask is not None:
            pyfits.append(fits_file_name, mask)

    def _baricenterCalculator(self, reference_image):
        ''' Calculate the peak position of the image
        args:
            reference_image = camera frame
        returns:
            cy, cx = y and x coord
        '''
        #scipy.signal.medfilt(reference_image, 3)
        counts, bin_edges = np.histogram(reference_image, bins=100)
        bin_edges=bin_edges[1:]
        thr = 5 * bin_edges[np.where(counts == max(counts))]
        idx = np.where(reference_image < thr)
        img = reference_image.copy()  
        #img[idx] = 0  
        cx, cy = centroids.centroid_com(img, mask=(img < thr))
        #cx, cy = centroids.centroid_2dg(img)
        #print(np.int(cy))
        #print(np.int(cx))
        return int(cy), int(cx)
    
    def _findBrightSpots(self, image, threshold, npixels=None):
        # Handle masked arrays
        if isinstance(image, np.ma.MaskedArray):
            mask = image.mask
            image = image.filled(0)  # Replace masked values with 0 to ignore them in processing
        else:
            mask = None
        
        # Convert 16-bit to 8-bit
        if image.dtype == np.uint16:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply threshold to detect bright spots
        _, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply mask if it exists
        if mask is not None:
            thresh[mask] = 0  # Ensure masked areas are ignored
        
        # Convert thresh to uint8 for findContours
        thresh_uint8 = thresh.astype(np.uint8)
        
        # Find contours of the bright regions
        contours, _ = cv2.findContours(thresh_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get the centroid of each bright spot
        bright_spots = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                brightness = int(np.max(image[cY-1:cY+2, cX-1:cX+2]))  # Convert to int to avoid overflow
                bright_spots.append((cX, cY, brightness))
        
        # Filter spots based on npixels separation
        if npixels:
            filtered_spots = []
            while bright_spots:
                # Sort by brightness and pick the brightest
                bright_spots.sort(key=lambda x: -x[2])
                brightest = bright_spots.pop(0)
                filtered_spots.append(brightest)
                
                # Remove spots within npixels distance from the selected brightest spot
                bright_spots = [spot for spot in bright_spots if np.linalg.norm(np.array(spot[:2]) - np.array(brightest[:2])) > npixels]
            
            bright_spots = [(x, y) for x, y, _ in filtered_spots]
        else:
            bright_spots = [(x, y) for x, y, _ in bright_spots]
        
        # print(bright_spots)
        return bright_spots
    
    def _preProcessing(self, image, cy, cx):
        ''' Cut the images around the pick position
        args:
            image = camera frame
        returns:
            crop = cut camera frame
        '''
        xcrop = config.CROP_WIDTH  # Using config value
        ycrop = config.CROP_HEIGHT  # Using config value

        if image.ndim != 2:
            raise ValueError("Expected a 2-dimensional image array")

        # Ensure image is float32 for calculations to save memory
        if image.dtype != np.float32:
             # Check if conversion is safe (e.g., from uint16)
             if np.can_cast(image.dtype, np.float32):
                 image = image.astype(np.float32)
             else:
                 # If cannot safely cast to float32, use float64
                 image = image.astype(np.float64)


        rows, cols = image.shape
        row_start, row_end = cy - ycrop, cy + ycrop
        col_start, col_end = cx - xcrop, cx + xcrop

        # Create a boolean mask: True for crop area (to be masked), False for background
        mask = np.zeros(image.shape, dtype=bool)
        # Clip indices to be within image bounds for safe slicing
        row_start_clip = max(0, row_start)
        row_end_clip = min(rows, row_end)
        col_start_clip = max(0, col_start)
        col_end_clip = min(cols, col_end)
        # Set mask to True only within the valid clipped region
        if row_start_clip < row_end_clip and col_start_clip < col_end_clip:
            mask[row_start_clip:row_end_clip, col_start_clip:col_end_clip] = True

        # Create a masked array where the crop area is masked (mask=True)
        masked_image = np.ma.array(image, mask=mask)

        # Calculate the mean of the *unmasked* (background) pixels.
        # Use the image's current dtype (hopefully float32) for the calculation.
        bkg = np.ma.mean(masked_image, dtype=image.dtype)
        if bkg is np.ma.masked: # Handle case where background is empty or all masked
            self._logger.warning("Background calculation resulted in a masked value. Using 0 as background.")
            bkg = 0
        else:
            # Ensure bkg is a scalar float of the correct type
            bkg = image.dtype.type(bkg)


        # Subtract background from the original image (result maintains dtype)
        img = image - bkg
        img = np.clip(img, 0, None) # Ensure non-negative values

        # --- Extract the crop with robust boundary handling ---
        # Create an empty array for the crop with the desired size and dtype
        crop = np.zeros((2 * ycrop, 2 * xcrop), dtype=img.dtype)

        # Calculate the valid source region within the image
        src_row_start = max(0, row_start)
        src_row_end = min(rows, row_end)
        src_col_start = max(0, col_start)
        src_col_end = min(cols, col_end)

        # Calculate the corresponding destination region within the crop array
        dst_row_start = src_row_start - row_start
        dst_row_end = src_row_end - row_start
        dst_col_start = src_col_start - col_start
        dst_col_end = src_col_end - col_start

        # Copy the valid data from img to crop only if regions are valid
        if (dst_row_end > dst_row_start) and (dst_col_end > dst_col_start) and \
           (src_row_end > src_row_start) and (src_col_end > src_col_start):
            crop[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = \
                img[src_row_start:src_row_end, src_col_start:src_col_end]
        # --- End crop extraction ---

        return crop
    
    def _capture_dark_frame(self, exptime):
        """
        Capture a dark frame with the lamp off.
        
        Args:
            exptime: Exposure time for the dark frame.
            
        Returns:
            dark_frame: The captured dark frame image.
        """
        self._logger.info("Turning off the filter to capture a dark frame...")
        self._filter.set_bandwidth_mode(1)  
        time.sleep(5) 
        self._set_camera_exposure_time(exptime)
        dark_frame_raw = self._camera.getFutureFrames(3).toNumpyArray()
        self._filter.set_bandwidth_mode(config.FILTER_BANDWIDTH_MODE) 
        time.sleep(config.FILTER_SETTLE_TIME_S)
        
        if dark_frame_raw.ndim == 3:
            dark_frame = np.mean(dark_frame_raw, axis=2).astype(np.float32) # Assuming (H,W,N)
        elif dark_frame_raw.ndim == 2:
             dark_frame = dark_frame_raw.astype(np.float32)
        else:
            self._logger.error(f"Unexpected dark frame dimensions: {dark_frame_raw.shape}")
            raise ValueError(f"Unexpected dark frame dimensions: {dark_frame_raw.shape}")

        self._logger.info(f"Dark frame shape: {dark_frame.shape}, dtype: {dark_frame.dtype}, min: {np.min(dark_frame)}, max: {np.max(dark_frame)}")
        return dark_frame

    def _save_dark_frame(self, dark_frame):
        """
        Save the captured dark frame with a timestamped filename.
        
        Args:
            dark_frame: The dark frame to save.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dark_frame_filename = os.path.join(self._dove, f"dark_{timestamp}.fits")
        self._logger.info(f"Saving dark frame as {dark_frame_filename}...")
        
        header = pyfits.Header()
        header['OBJECT'] = ('Dark Frame', 'Type of image')
        # Try to get actual exposure time from camera for the dark frame
        try:
            actual_dark_exptime = self._camera.getExposureTime()
            header['EXPTIME'] = (actual_dark_exptime, 'Exposure time in milliseconds')
        except Exception as e:
            self._logger.warning(f"Could not get camera exposure time for dark frame header: {e}")
            # Fallback to commanded exptime if actual can't be read (though it should have been just set)
            # header['EXPTIME'] = (self._exptime, 'Commanded exposure for dark series ms') 
        header['NFRAMES'] = (3, 'Number of frames averaged for dark')
        
        # Removed camera snapshot logic for dark frame header
        
        pyfits.writeto(dark_frame_filename, dark_frame.astype(np.float32), header=header, overwrite=True)

#     def _saveInfo(self, file_name):
#         fits_file_name = os.path.join(self._dove, file_name)
#         header = pyfits.Header()
#         header['PIX2UM'] = self._pix2um
#
#     @staticmethod
#     def reloadSplAcquirer():
#         theObject = SPLAquirer(None, None)
#         theObject._pix2um = header['PIX2UM']
