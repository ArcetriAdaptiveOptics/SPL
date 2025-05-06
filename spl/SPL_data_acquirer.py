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

    @staticmethod
    def _storageFolder(path=None):
        """ Creates the path where to save measurement data"""
        measurement_path = config.MEASUREMENT_ROOT_FOLDER
        return measurement_path
     
    def acquire(self, lambda_vector=None, exptime=None, numframes=None, mask=None, display_reference=False):
        """
        Acquires images at different wavelengths, subtracts dark frame, 
        and saves them as FITS cubes.
        
        Args:
            lambda_vector: Array of wavelengths to acquire. If None, uses config values
            exptime: Exposure time in milliseconds. If None, uses config value
            numframes: Number of frames to average. If None, uses config value
            mask: Optional mask to apply
            display_reference: If True, displays the reference image after dark subtraction
        """
        # Use configuration values if parameters are not provided
        if lambda_vector is None:
            lambda_vector = np.arange(config.LAMBDAMIN, config.LAMBDAMAX + 1, config.LAMBDASTEP)
        if exptime is None:
            exptime = config.EXPTIME
        if numframes is None:
            numframes = config.NUMFRAMES

        # Capture dark frame (with lamp off)
        dark_frame = self._capture_dark_frame(exptime)
        # Save the dark frame
        #self._save_dark_frame(dark_frame)
        
        # Now subtract the dark frame from subsequent frames during acquisition
        self._dark_frame = dark_frame  # Store the dark frame for later use

        self._dove, tt = tracking_number_folder.createFolderToStoreMeasurements(self._storageFolder())
        fits_file_name = os.path.join(self._dove, 'lambda_vector.fits')
        pyfits.writeto(fits_file_name, lambda_vector)

        # Step 1: Capture reference frame
        reference_image = self._captureReferenceFrame(config.REFERENCE_LAMBDA, exptime)
        
        # Ensure reference image is float32 and 2D
        if reference_image.ndim == 3:
            reference_image = np.mean(reference_image, axis=2)
        reference_image = reference_image.astype(np.float32)
        
        # Subtract the dark frame from the reference image
        reference_image -= self._dark_frame
        
        # Clip negative values to zero
        reference_image = np.clip(reference_image, 0, None)
        
        # Display reference image if requested
        if display_reference or config.SHOW_REFERENCE_FRAME:
            try:
                plt.figure(figsize=(10, 8))
                plt.imshow(reference_image, cmap='gray', origin='lower')
                plt.colorbar(label='Intensity')
                
                # --- Add Slice Boundary Visualization ---
                height, width = reference_image.shape
                n_rows_slice = self._n_rows
                n_cols_slice = self._m_cols
                row_size = height // n_rows_slice
                col_size = width // n_cols_slice

                # Draw horizontal lines
                for i in range(1, n_rows_slice):
                    plt.axhline(y=i * row_size, color='r', linestyle='--', alpha=0.7)
                # Draw vertical lines
                for j in range(1, n_cols_slice):
                    plt.axvline(x=j * col_size, color='r', linestyle='--', alpha=0.7)
                # --- End Slice Boundary Visualization ---

                plt.title(f'Reference Image (Dark Subtracted) with {n_rows_slice}x{n_cols_slice} Slice Boundaries')
                plt.xlabel("X pixel")
                plt.ylabel("Y pixel")
                plt.show()
            except Exception as e:
                self._logger.warning(f"Could not display reference image with slice boundaries: {e}")
        
        # Step 1.5: Slice the reference image and find spots in each subframe
        subframe_positions = self._sliceAndFindSpotPositions(reference_image, n=self._n_rows, m=self._m_cols)  
        print(f"Looking for spots in {len(subframe_positions)} subframes.")
        
        # Check if any spots were found in any subframe
        if not any(subframe_positions):
             # Raise error earlier if no spots are found anywhere
             raise ValueError("No bright spots detected in ANY subframe!")
        
        # Step 2: Combine spot positions from subframes and convert to full frame coordinates
        positions = []
        subframe_height = reference_image.shape[0] // self._n_rows
        subframe_width = reference_image.shape[1] // self._m_cols
        
        # Iterate through the flattened list of spot lists (one per subframe)
        for k, spots_in_subframe in enumerate(subframe_positions):
            if not spots_in_subframe:
                continue # Skip this subframe if no spots were found
                
            # Calculate the row and column index of this subframe
            row_idx = k // n_cols_slice
            col_idx = k % n_cols_slice
            
            # Iterate through spots found WITHIN this subframe
            for (cx_subframe, cy_subframe) in spots_in_subframe:
                # Convert subframe coordinates to full frame coordinates
                full_frame_y = row_idx * subframe_height + cy_subframe 
                full_frame_x = col_idx * subframe_width + cx_subframe
                positions.append((full_frame_x, full_frame_y))
        
        # Check if the final positions list is empty
        if not positions:
            raise ValueError("No valid spot coordinates generated after processing subframes!")
            
        print(f"Total spot positions identified: {len(positions)}")

        # Step 3: Scan through wavelengths and preprocess data
        frame_cube = []
        for wl in lambda_vector:
            print(f"Acquiring image at {wl} nm...")
            print(f'Exposure time: {exptime} ms')
            self._filter.move_to(wl)
            if wl == lambda_vector[0]:
                time.sleep(config.FILTER_SETTLE_TIME_S)
            self._camera.setExposureTime(exptime)
            image = self._camera.getFutureFrames(numframes).toNumpyArray()

            # If the image has more than one frame, average them
            if image.ndim == 3:
                image = np.mean(image, axis=2)
            else:
                image = np.mean(image, axis=0)

            # Subtract the dark frame from the acquired image
            image -= self._dark_frame

            # Clip negative values to zero
            image = np.clip(image, 0, None)

            # Step 4: Pre-process each frame for all detected positions
            processed_wavelength_frames = []
            for idx, pos in enumerate(positions):
                cx, cy = pos
                crop = self._preProcessing(image, cy, cx)
                file_name = 'image_%dnm_pos%02d.fits' % (wl, idx)
                self._saveCameraFrame(file_name, crop, wavelength=wl)
                processed_wavelength_frames.append(crop)
            
            frame_cube.append(processed_wavelength_frames)

        print(f"Data saved in {self._dove}")
        print("Acquisition complete.")
        return tt

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
        print(f'Acquiring reference image at {wavelength} nm...')
        print(f'Exposure time: {exptime} ms')
        self._filter.move_to(wavelength)
        time.sleep(config.FILTER_SETTLE_TIME_S)  # Using config value
        self._camera.setExposureTime(exptime)
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

    def _saveCameraFrame(self, file_name, frame, wavelength=None):
        ''' Save camera frames in SPL/TrackingNumber
        '''
        fits_file_name = os.path.join(self._dove, file_name)
        if isinstance(frame, np.ma.MaskedArray):
            data = frame.data
            mask = frame.mask.astype(np.int32)
        else:
            data = frame
            mask = None

        # Create header with wavelength information if provided
        header = pyfits.Header()
        if wavelength is not None:
            header['WAVELEN'] = (wavelength, 'Wavelength in nanometers')

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
        print("Turning off the filter to capture a dark frame...")
        self._filter.set_bandwidth_mode(1)  
        time.sleep(5)
        self._camera.setExposureTime(exptime)
        dark_frame = self._camera.getFutureFrames(1).toNumpyArray()
        self._filter.set_bandwidth_mode(2)
        
        # Convert to a 2D array and ensure float32 type
        if dark_frame.ndim == 3:
            dark_frame = np.mean(dark_frame, axis=2)
        dark_frame = dark_frame.astype(np.float32)
        print(f"Dark frame shape: {dark_frame.shape}, dtype: {dark_frame.dtype}, min: {np.min(dark_frame)}, max: {np.max(dark_frame)}")
        return dark_frame

    def _save_dark_frame(self, dark_frame):
        """
        Save the captured dark frame with a timestamped filename.
        
        Args:
            dark_frame: The dark frame to save.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dark_frame_filename = os.path.join(self._dove, f"dark_{timestamp}.fits")
        print(f"Saving dark frame as {dark_frame_filename}...")
        
#     def _saveInfo(self, file_name):
#         fits_file_name = os.path.join(self._dove, file_name)
#         header = pyfits.Header()
#         header['PIX2UM'] = self._pix2um
#
#     @staticmethod
#     def reloadSplAcquirer():
#         theObject = SPLAquirer(None, None)
#         theObject._pix2um = header['PIX2UM']
