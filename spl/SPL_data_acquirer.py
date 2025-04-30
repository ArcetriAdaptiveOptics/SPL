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
        self._pix2um = 4.5 # pixel to um conversion for AVT GT5120
        self._exptime = None
        self._n_rows = config.N_ROWS
        self._m_cols = config.M_COLS

    @staticmethod
    def _storageFolder(path=None):
        """ Creates the path where to save measurement data"""
        measurement_path = config.MEASUREMENT_ROOT_FOLDER
        return measurement_path
     
    def acquire(self, lambda_vector, exptime, numframes=2, mask=None, display_reference=False):
        """
        Acquires images at different wavelengths, subtracts dark frame, 
        and saves them as FITS cubes.
        
        Args:
            lambda_vector: Array of wavelengths to acquire
            exptime: Exposure time in milliseconds
            numframes: Number of frames to average (default: 2)
            mask: Optional mask to apply
            display_reference: If True, displays the reference image after dark subtraction (default: False)
        """
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
        reference_image = self._captureReferenceFrame(700, exptime)
        
        # Ensure reference image is float32 and 2D
        if reference_image.ndim == 3:
            reference_image = np.mean(reference_image, axis=2)
        reference_image = reference_image.astype(np.float32)
        #print(f"Reference image shape: {reference_image.shape}, dtype: {reference_image.dtype}, min: {np.min(reference_image)}, max: {np.max(reference_image)}")
        
        # Subtract the dark frame from the reference image
        reference_image -= self._dark_frame
        #print(f"After dark subtraction - min: {np.min(reference_image)}, max: {np.max(reference_image)}")
        
        # Clip negative values to zero
        reference_image = np.clip(reference_image, 0, None)
        
        # Display reference image if requested
        if display_reference:
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
        subframe_height = reference_image.shape[0] // 3
        subframe_width = reference_image.shape[1] // 2
        n_cols_slice = 2 # Number of columns used in slicing
        
        # Iterate through the flattened list of spot lists (one per subframe)
        for k, spots_in_subframe in enumerate(subframe_positions):
            if not spots_in_subframe:
                # Optional: Log if a subframe is empty
                # self._logger.debug(f"No spots found in subframe index {k}")
                continue # Skip this subframe if no spots were found
                
            # Calculate the row and column index of this subframe
            row_idx = k // n_cols_slice
            col_idx = k % n_cols_slice
            
            # Iterate through spots found WITHIN this subframe
            for (cx_subframe, cy_subframe) in spots_in_subframe: # Assuming spots are (x, y)
                # Convert subframe coordinates (relative to subframe top-left) 
                # to full frame coordinates (relative to full frame top-left)
                # cy_subframe = row within subframe, cx_subframe = col within subframe
                # Check coordinate order consistency with _findBrightSpots if issues arise
                full_frame_y = row_idx * subframe_height + cy_subframe 
                full_frame_x = col_idx * subframe_width + cx_subframe
                positions.append((full_frame_x, full_frame_y)) # Append as (x, y) tuple
        
        # Check if the final positions list is empty (could happen if spots lists were empty)
        if not positions:
            raise ValueError("No valid spot coordinates generated after processing subframes!")
            
        print(f"Total spot positions identified: {len(positions)}")

        # Step 3: Scan through wavelengths and preprocess data (looping over positions)
        frame_cube = []  # This will hold the final 4D cube (npix, mpix, nwaves, npositions)
        for wl in lambda_vector:
            print(f"Acquiring image at {wl} nm...")
            print(f'Exposure time: {exptime} ms')
            self._filter.move_to(wl)
            if wl == lambda_vector[0]:
                time.sleep(10)
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
            processed_wavelength_frames = []  # Holds preprocessed images for each position at this wavelength
            for idx, pos in enumerate(positions):
                cx, cy = pos  # Spot coordinates (now full frame coordinates)
                #print(f"Processing spot at (cx={cx}, cy={cy}) at {wl} nm")
                crop = self._preProcessing(image, cy, cx)  # Preprocess image around the spot
                file_name = 'image_%dnm_pos%02d.fits' % (wl, idx)
                self._saveCameraFrame(file_name, crop, wavelength=wl)
                processed_wavelength_frames.append(crop)
            
            # Append the processed frames for the current wavelength (shape: npix, mpix, npositions)
            frame_cube.append(processed_wavelength_frames)

        # Convert the frame_cube to a numpy array of shape (npix, mpix, nwaves, npositions)
        #frame_cube = np.array(frame_cube)
        
        # Step 5: Save data
        #self._saveFitsCubes(lambda_vector, frame_cube)
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
        time.sleep(10)
        self._camera.setExposureTime(exptime)
        image = self._camera.getFutureFrames(1).toNumpyArray()
        # plt.imshow(image, cmap='gray')
        # plt.title(f'Reference Image at {wavelength} nm')
        # plt.show()
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
        xcrop = 150
        #xcrop = 100
        ycrop = 100

        if image.ndim != 2:
            raise ValueError("Expected a 2-dimensional image array")

        tmp = np.zeros((image.shape[0], image.shape[1]))
        tmp[cy-ycrop: cy+ycrop, cx-xcrop: cx+xcrop] = 1
        id_bkg = np.where(tmp == 0)
        bkg = np.ma.mean(image[id_bkg])
        img = image - bkg
        crop = img[cy-ycrop: cy+ycrop, cx-xcrop: cx+xcrop]
        # print('Crop peak counts', int(np.max(crop)))
        # plt.imshow(crop, cmap='gray')
        # plt.show()
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
