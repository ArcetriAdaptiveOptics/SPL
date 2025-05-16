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
from tqdm import tqdm # Added for progress bar
from spl.conf import configuration # Import configuration
import re
from spl import process_psf
import matplotlib.pyplot as plt

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
        self._crop_size = 150 #configuration.FRINGES_HEIGHT  # Use configuration value

    @staticmethod
    def _storageFolder():
        """ Creates the path where to save measurement data"""
        return configuration.MEASUREMENT_ROOT_FOLDER # Use configuration value

    @staticmethod
    def _parse_rotation_angles_from_config(angle_string: str | None) -> dict[int, float]:
        """Parses a string like '0:15.0,1:-5.5' into {0: 15.0, 1: -5.5}. Handles None input."""
        angles: dict[int, float] = {}
        if not angle_string:
            return angles
        try:
            pairs = angle_string.split(',')
            for pair in pairs:
                if ':' not in pair:
                    raise ValueError(f"Invalid pair format: '{pair}'")
                pos_str, angle_str = pair.split(':', 1)
                pos = int(pos_str.strip())
                angle = float(angle_str.strip())
                angles[pos] = angle
            logging.info(f"Parsed config rotation angles: {angles}")
            return angles
        except ValueError as e:
            logging.error(f"Error parsing config rotation angles string '{angle_string}': {e}")
            # Consider if re-raising is appropriate or returning empty dict / default
            raise # Re-raise for now, consistent with original parse_position_angles

    def process(self, tt: str, debug_contours: bool = False):
        '''
        Process measurement data by fitting ellipses and rotating frames.

        Parameters
        ----------
        tt: string
            tracking number of the measurement data
        debug_contours: bool
            If True, save debug PNGs showing contours used for mean image centroid.
        '''
        # Parse rotation angles from configuration
        config_rotation_angles = self._parse_rotation_angles_from_config(configuration.POSITIONS_TO_ROTATION_ANGLES)
        
        dove = os.path.join(self._storageFolder(), tt)
            
        self._logger.info('Processing data with tt = %s (Debug Contours: %s, Config Angles: %s)', 
                        tt, debug_contours, config_rotation_angles if config_rotation_angles else "None/Empty")

        # Extract wavelengths and positions
        wavelengths, positions = self.parseFitsFilenames(dove)
        
        # Filter positions based on configuration if positions_to_rotation_angles is provided
        if config_rotation_angles:
            filtered_positions = [pos for pos in positions if pos in config_rotation_angles]
            if not filtered_positions:
                self._logger.warning("No positions found matching those specified in positions_to_rotation_angles. Skipping processing.")
                return
            self._logger.info(f"Filtering positions to only process: {filtered_positions}")
            positions = filtered_positions
        
        # Process each position
        for position in tqdm(positions, desc="Processing Positions", unit="pos"):
            self._processPosition(dove, position, wavelengths, debug_contours, config_rotation_angles)

    def parseFitsFilenames(self, data_folder: str) -> tuple[list[int], list[int]]:
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
        
        return sorted(list(wavelengths)), sorted(list(positions))

    def _processPosition(self, folder: str, position: int, wavelengths: list[int], 
                         debug_contours: bool, config_rotation_angles: dict[int, float]):
        """Process all frames for a specific position using a common rotation angle around geometric center."""
        self._logger.info(f"--- Processing Position {position:02d} ---")
        
        # 1. Load images and Calculate ALL Individual Centroids first
        path_list = sorted(glob.iglob(os.path.join(folder, f'image_*nm_pos{position:02d}.fits')))
        if not path_list:
            self._logger.error(f"No raw FITS files found for position {position:02d}. Skipping.")
            return
            
        frame_data = [] # Store tuples: (raw_image, header, path, wavelength)

        for i, path in enumerate(tqdm(path_list, desc=f"Frames Pos {position:02d}", unit="frame", leave=False)):
            current_wavelength = wavelengths[i]
            try:
                with pyfits.open(path) as hduList:
                    image = hduList[0].data
                    if image is None:
                         self._logger.warning(f"No data found in {path}. Skipping file.")
                         continue
                    raw_image = image.astype(np.float32)
                    header = hduList[0].header
                    frame_data.append((raw_image, header, path, current_wavelength))

            except Exception as e:
                self._logger.error(f"Error reading {path}: {e}. Skipping file.")
                
        if not frame_data:
             self._logger.error(f"No valid images loaded for position {position:02d}. Skipping position.")
             return

        # 2. Determine Common Rotation Angle from Mean Image
        mean_image = None
        try:
            all_raw_images = [fd[0] for fd in frame_data] # Extract raw images
            if all_raw_images:
                mean_image = np.mean(np.stack(all_raw_images, axis=0), axis=0)
                self._logger.info(f"Calculated mean image for rotation reference (shape: {mean_image.shape})")
        except Exception as e:
             self._logger.error(f"Could not stack images to calculate mean image: {e}")
             return

        common_angle = 0.0
        angle_source = "Default (0.0)"
        calculated_angle = 0.0 # Initialize in case common_properties is None

        if mean_image is not None:
            try:
                debug_prefix_mean = os.path.join(folder, f"pos{position:02d}_mean_image")
                common_properties = process_psf.calculate_psf_properties(mean_image, debug_contours, debug_prefix_mean)
                
                if common_properties:
                    calculated_angle, _ = common_properties # We only need the angle, not the centroid
                    self._logger.info(f"Properties from mean image: Angle={calculated_angle:.2f}")

                # Determine common_angle based on config, then calculated, then default
                if position in config_rotation_angles:
                    common_angle = config_rotation_angles[position]
                    angle_source = f"Config ({common_angle:.2f} deg)"
                elif common_properties: # Only use calculated_angle if properties were found
                    common_angle = calculated_angle
                    angle_source = f"Calculated ({common_angle:.2f} deg)"
                else: # Fallback if no config and no calculated properties
                    common_angle = 0.0 # Already initialized, but explicit for clarity
                    angle_source = f"Default (0.0 deg - no config/calc)"

            except Exception as e:
                self._logger.error(f"Error calculating mean image properties or determining angle: {e}. Using angle=0.")
                common_angle = 0.0
                angle_source = "Error/Default (0.0 deg)"
        else: # Mean image is None
             self._logger.error("Mean image is None, cannot determine common properties.")
             return
             
        self._logger.info(f"Common Angle for Rotation: {common_angle:.2f} (Source: {angle_source})")
        
        # 3. Process each individual frame: Rotate around geometric center
        for raw_image, header, path, current_wavelength in frame_data:
            self._logger.info(f"Processing frame {os.path.basename(path)} (Wavelength: {current_wavelength} nm)")
            
            # Calculate geometric center of the image
            height, width = raw_image.shape
            geometric_center = np.array([(height - 1) / 2.0, (width - 1) / 2.0])
            
            # Pass the raw_image to _processFrame for rotation and cropping
            try:
                processed_image = self._processFrame(raw_image, common_angle, geometric_center, debug_contours, path)
                
                if processed_image is None: 
                    self._logger.warning(f"Skipping save for frame {os.path.basename(path)} due to processing error in _processFrame.")
                    continue

                # Save the processed image
                new_filename = path.replace('.fits', '_proc.fits')
                # Update header
                header['HISTORY'] = f"Processed by SPL_data_processer."
                header['HISTORY'] = f"Applied common rotation angle: {common_angle:.2f} deg around geometric center"
                header['HISTORY'] = f"Rotation angle source: {angle_source}"
                
                pyfits.writeto(new_filename, processed_image, header=header, overwrite=True)
                self._logger.info(f"Saved processed image to {new_filename}")
                
            except Exception as e:
                self._logger.error(f"Error processing/saving frame from {path}: {e}")

        # --- Debug Plots --- (if debug_contours is True)
        if debug_contours:
            try:
                plt.figure(figsize=(10, 6))
                plt.imshow(mean_image, cmap='gray', origin='lower')
                plt.colorbar(label='Intensity')
                plt.title(f'Position {position:02d}: Mean Image')
                plt.xlabel("X pixel")
                plt.ylabel("Y pixel")
                plot_filename = os.path.join(folder, f"pos{position:02d}_mean_image.png")
                plt.savefig(plot_filename)
                plt.close()
                self._logger.info(f"Saved mean image debug plot to {plot_filename}")
            except Exception as e:
                self._logger.warning(f"Could not generate or save mean image debug plot for position {position:02d}: {e}")

        self._logger.info(f"--- Finished Position {position:02d} ---")
        
    def _processFrame(self, image_to_rotate, common_angle, common_centroid, debug_contours, original_path):
        """Process a single frame (already Y-panned) using common rotation around common centroid, then cropping."""
        
        # 1. Rotate image around the COMMON centroid using the COMMON angle
        rotated_image = image_to_rotate # Start with potentially shifted image
        if abs(common_angle) > 1e-6:
            self._logger.info(f"  Applying common rotation: {common_angle:.2f} deg around common centroid ({common_centroid[0]:.2f}, {common_centroid[1]:.2f})")
            try:
                # Use the rotation function from process_psf
                rotated_image = process_psf.rotate_around_centroid(image_to_rotate, common_angle, common_centroid)
            except Exception as e:
                self._logger.error(f"  Rotation failed: {e}. Returning unrotated (but possibly shifted) image.")
                # Keep image_to_rotate as the fallback
        else:
            self._logger.info("  Skipping rotation (common angle is ~0).")
        
        # 2. Crop the rotated image around its geometric center
        image_to_crop = rotated_image 
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

        # 3. Pad if necessary to reach target crop size
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
            
        # --- Debug: Save final processed image --- 
        if debug_contours:
             debug_final_png_path = original_path.replace('.fits', '_proc_debug.png')
             process_psf.save_float_image_as_png(cropped, debug_final_png_path)
        # --- End Debug ---
        
        return cropped
