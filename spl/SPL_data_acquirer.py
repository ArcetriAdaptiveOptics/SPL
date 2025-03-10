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

    @staticmethod
    def _storageFolder(path=None):
        """ Creates the path where to save measurement data"""
        measurement_path = config.MEASUREMENT_ROOT_FOLDER
        return measurement_path

    def acquire(self, lambda_vector, exptime, numframes=2, mask=None):
        """
        Acquires images at different wavelengths and saves them as FITS cubes.
        """

        self._dove, tt = tracking_number_folder.createFolderToStoreMeasurements(self._storageFolder())
        fits_file_name = os.path.join(self._dove, 'lambda_vector.fits')
        pyfits.writeto(fits_file_name, lambda_vector)

        # Step 1: Capture reference frame
        reference_image = self._captureReferenceFrame(700, exptime*2)
        
        # Step 1.5: Slice the reference image and find spots in each subframe
        subframe_positions = self._sliceAndFindSpotPositions(reference_image, n=3, m=2)  # Example: 3x2 subframes
        print(f"Looking for spots in subframes: {len(subframe_positions)} positions detected.")
        
        if not subframe_positions:
            raise ValueError("No bright spots detected in any subframe!")
        
        # Step 2: Combine spot positions from subframes and convert to full frame coordinates
        positions = []
        subframe_height, subframe_width = reference_image.shape[0] // 3, reference_image.shape[1] // 2  # Assuming 3x2 subframes
        
        for i, subframe in enumerate(subframe_positions):
            for j, (cx_subframe, cy_subframe) in enumerate(subframe):
                # Convert subframe coordinates to full frame coordinates
                full_cx = i * subframe_width + cx_subframe
                full_cy = j * subframe_height + cy_subframe
                positions.append((full_cx, full_cy))
        
        print(f"Total found spots: {len(positions)}")

        # Step 3: Scan through wavelengths and preprocess data (looping over positions)
        frame_cube = []  # This will hold the final 4D cube (npix, mpix, nwaves, npositions)
        for wl in lambda_vector:
            print(f"Acquiring image at {wl} nm...")
            print(f'Exposure time: {exptime} ms')
            self._filter.move_to(wl)
            self._camera.setExposureTime(exptime)
            image = self._camera.getFutureFrames(numframes).toNumpyArray()

            # If the image has more than one frame, average them
            if image.ndim == 3:
                image = np.mean(image, axis=2)
            else:
                image = np.mean(image, axis=0)

            # Step 4: Pre-process each frame for all detected positions
            processed_wavelength_frames = []  # Holds preprocessed images for each position at this wavelength
            for idx, pos in enumerate(positions):
                cx, cy = pos  # Spot coordinates (now full frame coordinates)
                #print(f"Processing spot at (cx={cx}, cy={cy}) at {wl} nm")
                crop = self._preProcessing(image, cy, cx)  # Preprocess image around the spot
                file_name = 'image_%dnm_pos%02d.fits' % (wl, idx)
                self._saveCameraFrame(file_name, crop)
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

    def _saveCameraFrame(self, file_name, frame):
        ''' Save camera frames in SPL/TrackingNumber
        '''
        fits_file_name = os.path.join(self._dove, file_name)
        if isinstance(frame, np.ma.MaskedArray):
            data = frame.data
            mask = frame.mask.astype(np.int32)
        else:
            data = frame
            mask = None

        pyfits.writeto(fits_file_name, data, overwrite=True)
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
        
        # Find contours of the bright regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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
        
#     def _saveInfo(self, file_name):
#         fits_file_name = os.path.join(self._dove, file_name)
#         header = pyfits.Header()
#         header['PIX2UM'] = self._pix2um
#
#     @staticmethod
#     def reloadSplAcquirer():
#         theObject = SPLAquirer(None, None)
#         theObject._pix2um = header['PIX2UM']
