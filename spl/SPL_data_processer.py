'''
Authors
  - M. Bonaglia: created in 2025
'''

import os
import numpy as np
import glob
import logging
from astropy.io import fits as pyfits
from scipy.ndimage import rotate, shift 
from spl.conf import configuration as config
import re
from spl import process_psf

class SplProcessor():
    '''
    Class used to process SPL camera images by fitting ellipses, rotating and centering them.

    HOW TO USE IT::
        from spl import SPL_data_processer import SplProcessor
        proc = SplProcessor()
        proc.process(tt)
    '''

    def __init__(self):
        """The constructor """
        self._logger = logging.getLogger('SPL_PROC:')
        self._crop_size = 150  # Size of the cropped region around the spot

    @staticmethod
    def _storageFolder():
        """ Creates the path where to save measurement data"""
        measurement_path = config.MEASUREMENT_ROOT_FOLDER
        return measurement_path

    def process(self, tt, do_pan=True):
        '''
        Process measurement data by fitting ellipses, rotating and centering frames.

        Parameters
        ----------
        tt: string
            tracking number of the measurement data
        do_pan: bool
            If True (default), pan the image to center the calculated centroid.
        '''
        dove = os.path.join(self._storageFolder(), tt)
        self._logger.info('Processing data with tt = %s (Panning: %s)', tt, do_pan)

        # Extract wavelengths and positions
        wavelengths, positions = self.parseFitsFilenames(dove)
        
        # Process each position
        for position in positions:
            self._processPosition(dove, position, do_pan)

    def parseFitsFilenames(self, data_folder):
        """Parse FITS filenames to extract wavelengths and positions."""
        pattern = re.compile(r"image_(\d{3})nm_pos(\d{2})\.fits")
        wavelengths = set()
        positions = set()
        
        for filename in os.listdir(data_folder):
            match = pattern.match(filename)
            if match:
                wavelength, position = match.groups()
                wavelengths.add(int(wavelength))
                positions.add(int(position))
        
        return sorted(wavelengths), sorted(positions)

    def _processPosition(self, folder, position, do_pan):
        """Process all frames for a specific position using a common rotation angle."""
        self._logger.info(f"--- Processing Position {position:02d} ---")
        
        # 1. Load all raw images for this position
        path_list = sorted(glob.iglob(os.path.join(folder, f'image_*nm_pos{position:02d}.fits')))
        if not path_list:
            self._logger.error(f"No raw FITS files found for position {position:02d}. Skipping.")
            return
            
        images_data = []
        headers_data = []
        for path in path_list:
            try:
                with pyfits.open(path) as hduList:
                    image = hduList[0].data
                    if image is None:
                         self._logger.warning(f"No data found in {path}. Skipping file.")
                         continue
                    images_data.append(image.astype(np.float32))
                    headers_data.append(hduList[0].header)
            except Exception as e:
                self._logger.error(f"Error reading {path}: {e}. Skipping file.")
                
        if not images_data:
             self._logger.error(f"No valid images loaded for position {position:02d}. Skipping position.")
             return

        # 2. Calculate median image and estimate common rotation angle
        try:
            median_image = np.median(np.stack(images_data, axis=0), axis=0)
            self._logger.info(f"Calculated median image for position {position:02d} (shape: {median_image.shape})")
        except Exception as e:
            self._logger.error(f"Error calculating median image for position {position:02d}: {e}. Skipping position.")
            return

        common_properties = process_psf.calculate_psf_properties(median_image)
        common_angle = 0.0 # Default if calculation fails
        if common_properties:
            common_angle, _ = common_properties # We only need the angle from the median
            self._logger.info(f"Estimated common rotation angle for position {position:02d}: {common_angle:.2f} degrees")
        else:
            self._logger.warning(f"Could not estimate common rotation angle from median image for position {position:02d}. Using 0 degrees.")

        # 3. Process each individual frame using the common angle
        for i, (raw_image, header, path) in enumerate(zip(images_data, headers_data, path_list)):
            self._logger.info(f"Processing frame {i+1}/{len(images_data)} from {os.path.basename(path)}")
            try:
                # Process the image, passing common angle and do_pan flag
                processed_image = self._processFrame(raw_image, common_angle, do_pan)
                
                # Save the processed image (add history?)
                # TODO: Consider adding common angle and pan status to header here?
                new_filename = path.replace('.fits', '_proc.fits')
                pyfits.writeto(new_filename, processed_image, header=header, overwrite=True)
                self._logger.info(f"Saved processed image to {new_filename}")
                
            except Exception as e:
                self._logger.error(f"Error processing frame from {path}: {e}")
        
        self._logger.info(f"--- Finished Position {position:02d} ---")
        
    def _processFrame(self, image_float, common_angle, do_pan):
        """Process a single frame using a common rotation angle, optionally panning, and cropping."""
        # Ensure image is float
        image = image_float.astype(np.float32)

        # 1. Rotate image using the COMMON calculated angle
        rotated_image = image # Start with original if angle is 0
        if abs(common_angle) > 1e-6:
            self._logger.info(f"  Applying common rotation: {common_angle:.2f} degrees.")
            rotated_image = rotate(image, common_angle, reshape=False, order=3, mode='constant', cval=0.0)
        else:
            self._logger.info("  Skipping rotation (common angle is ~0).")
        
        # 2. Optionally Pan image to center
        image_to_crop = rotated_image # Default to rotated image if not panning
        if do_pan:
            # Calculate centroid for THIS SPECIFIC FRAME (before rotation) for accurate panning
            individual_properties = process_psf.calculate_psf_properties(image) 
            if individual_properties:
                _, individual_centroid = individual_properties
                self._logger.info(f"  Individual centroid for panning: ({individual_centroid[0]:.2f}, {individual_centroid[1]:.2f})")
                panned_image, applied_shift = process_psf.pan_image_to_center(rotated_image, individual_centroid)
                self._logger.info(f"  Panned image. Applied shift (row, col): ({applied_shift[0]:.2f}, {applied_shift[1]:.2f})")
                image_to_crop = panned_image
            else:
                 self._logger.warning("  Panning requested but failed to find individual centroid. Cropping rotated image.")
        else:
            self._logger.info("  Skipping panning step.")
            
        # 3. Crop the image_to_crop (either panned or just rotated) around its geometric center
        height, width = image_to_crop.shape
        center_y = (height - 1) / 2.0
        center_x = (width - 1) / 2.0
        
        # Calculate crop boundaries (integer indices needed)
        half_crop_y = self._crop_size // 2
        half_crop_x = self._crop_size // 2
        
        y_start = max(0, int(round(center_y - half_crop_y)))
        y_end = min(height, int(round(center_y + half_crop_y)))
        x_start = max(0, int(round(center_x - half_crop_x)))
        x_end = min(width, int(round(center_x + half_crop_x)))
            
        # Crop the image
        cropped = image_to_crop[y_start:y_end, x_start:x_end]
        self._logger.info(f"  Cropped image to shape: {cropped.shape}")

        # 4. Pad if necessary to reach target crop size
        if cropped.shape[0] < self._crop_size or cropped.shape[1] < self._crop_size:
            pad_y_total = self._crop_size - cropped.shape[0]
            pad_x_total = self._crop_size - cropped.shape[1]
            # Distribute padding (approximately) evenly
            pad_y_before = pad_y_total // 2
            pad_y_after = pad_y_total - pad_y_before
            pad_x_before = pad_x_total // 2
            pad_x_after = pad_x_total - pad_x_before
            
            cropped = np.pad(cropped, 
                           ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after)), 
                           mode='constant', constant_values=0.0)
            self._logger.info(f"  Padded cropped image to final shape: {cropped.shape}")
            
        return cropped
