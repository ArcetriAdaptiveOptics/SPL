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
        self._crop_size = 150  # Size of the cropped region around the spot

    @staticmethod
    def _storageFolder():
        """ Creates the path where to save measurement data"""
        measurement_path = config.MEASUREMENT_ROOT_FOLDER
        return measurement_path

    def process(self, tt, debug_contours=False, position_angles=None):
        '''
        Process measurement data by fitting ellipses and rotating frames.

        Parameters
        ----------
        tt: string
            tracking number of the measurement data
        debug_contours: bool
            If True, save debug PNGs showing contours used for mean image centroid.
        position_angles: dict, optional
            Dictionary mapping position index (int) to predefined rotation angle (float, degrees).
            Overrides automatic angle calculation for specified positions. Defaults to None.
        '''
        dove = os.path.join(self._storageFolder(), tt)
        # Ensure position_angles is a dict
        if position_angles is None:
            position_angles = {}
            
        self._logger.info('Processing data with tt = %s (Debug Contours: %s, Angles: %s)', 
                        tt, debug_contours, position_angles if position_angles else "Auto")

        # Extract wavelengths and positions
        wavelengths, positions = self.parseFitsFilenames(dove)
        
        # Process each position
        for position in positions:
            self._processPosition(dove, position, wavelengths, debug_contours, position_angles)

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
        
        return sorted(list(wavelengths)), sorted(list(positions))

    def _processPosition(self, folder, position, wavelengths, debug_contours, position_angles):
        """Process all frames for a specific position using a common rotation angle and centroid, with pre-rotation Y-panning based on linear fit."""
        self._logger.info(f"--- Processing Position {position:02d} ---")
        
        # 1. Load images and Calculate ALL Individual Centroids first
        path_list = sorted(glob.iglob(os.path.join(folder, f'image_*nm_pos{position:02d}.fits')))
        if not path_list:
            self._logger.error(f"No raw FITS files found for position {position:02d}. Skipping.")
            return
            
        frame_data = [] # Store tuples: (raw_image, header, path, wavelength, centroid_y, centroid_x)
        valid_wavelengths_for_fit = []
        valid_centroid_y_for_fit = []

        for i, path in enumerate(path_list):
            current_wavelength = wavelengths[i]
            try:
                with pyfits.open(path) as hduList:
                    image = hduList[0].data
                    if image is None:
                         self._logger.warning(f"No data found in {path}. Skipping file.")
                         continue
                    raw_image = image.astype(np.float32)
                    header = hduList[0].header
                    
                    # Calculate individual centroid
                    individual_props = process_psf.calculate_psf_properties(raw_image, debug_save_contour=False)
                    centroid_y, centroid_x = np.nan, np.nan # Defaults
                    if individual_props:
                        _, individual_centroid = individual_props
                        centroid_y = individual_centroid[0] # Y coordinate (row)
                        centroid_x = individual_centroid[1] # X coordinate (col)
                        valid_wavelengths_for_fit.append(current_wavelength)
                        valid_centroid_y_for_fit.append(centroid_y)
                    else:
                        self._logger.warning(f"Could not calculate centroid for {os.path.basename(path)}.")
                        
                    frame_data.append((raw_image, header, path, current_wavelength, centroid_y, centroid_x))

            except Exception as e:
                self._logger.error(f"Error reading or getting centroid for {path}: {e}. Skipping file.")
                
        if not frame_data:
             self._logger.error(f"No valid images loaded or centroids calculated for position {position:02d}. Skipping position.")
             return
        # 2.1 Perform Weighted Linear Fit for X-Centroid vs Wavelength
        x_fit_coeffs = None
        if len(valid_wavelengths_for_fit) >= 2: # Need at least 2 points for a line fit
            try:
                x_fit = np.array(valid_wavelengths_for_fit)
                x_centroid_fit = np.array([fd[5] for fd in frame_data if not np.isnan(fd[5])]) # Extract X centroids
                
                # --- Calculate Gaussian Weights --- 
                center_wl = (x_fit.min() + x_fit.max()) / 2.0
                # Use 1/4 of the wavelength range as sigma for weighting
                sigma_wl = (x_fit.max() - x_fit.min()) / 4.0 
                if sigma_wl < 1e-6: sigma_wl = 1.0 # Avoid division by zero if range is tiny
                weights = np.exp(-((x_fit - center_wl)**2) / (2 * sigma_wl**2))
                # --- End Weights --- 
                
                x_fit_coeffs = np.polyfit(x_fit, x_centroid_fit, deg=1, w=weights)
                self._logger.info(f"Weighted linear fit (X vs Wavelength): Slope={x_fit_coeffs[0]:.4f}, Intercept={x_fit_coeffs[1]:.2f}")
            except Exception as e:
                self._logger.error(f"Error performing weighted linear fit for X-centroid: {e}. Panning will be skipped.")
                x_fit_coeffs = None
        else:
            self._logger.warning("Not enough valid centroid points (<2) to perform linear fit. Panning will be skipped.")

        # 2. Perform Weighted Linear Fit for Y-Centroid vs Wavelength
        y_fit_coeffs = None
        if len(valid_wavelengths_for_fit) >= 2: # Need at least 2 points for a line fit
            try:
                x_fit = np.array(valid_wavelengths_for_fit)
                y_fit = np.array(valid_centroid_y_for_fit)
                
                # --- Calculate Gaussian Weights --- 
                center_wl = (x_fit.min() + x_fit.max()) / 2.0
                # Use 1/4 of the wavelength range as sigma for weighting
                sigma_wl = (x_fit.max() - x_fit.min()) / 4.0 
                if sigma_wl < 1e-6: sigma_wl = 1.0 # Avoid division by zero if range is tiny
                weights = np.exp(-((x_fit - center_wl)**2) / (2 * sigma_wl**2))
                # --- End Weights --- 
                
                y_fit_coeffs = np.polyfit(x_fit, y_fit, deg=1, w=weights)
                self._logger.info(f"Weighted linear fit (Y vs Wavelength): Slope={y_fit_coeffs[0]:.4f}, Intercept={y_fit_coeffs[1]:.2f}")
            except Exception as e:
                self._logger.error(f"Error performing weighted linear fit for Y-centroid: {e}. Panning will be skipped.")
                y_fit_coeffs = None
        else:
            self._logger.warning("Not enough valid centroid points (<2) to perform linear fit. Panning will be skipped.")

        # 3. Determine Common Rotation Angle and Centroid from Mean Image (still needed for rotation reference)
        # Calculate mean image using the raw_images stored in frame_data
        mean_image = None
        try:
            all_raw_images = [fd[0] for fd in frame_data] # Extract raw images
            if all_raw_images:
                mean_image = np.mean(np.stack(all_raw_images, axis=0), axis=0)
                self._logger.info(f"Calculated mean image for rotation reference (shape: {mean_image.shape})")
        except Exception as e:
             self._logger.error(f"Could not stack images to calculate mean image: {e}")
             # Cannot proceed without mean image properties
             return

        common_angle = 0.0
        common_centroid = None
        angle_source = "Default (0.0)"
        if mean_image is not None:
            try:
                debug_prefix_mean = os.path.join(folder, f"pos{position:02d}_mean_image")
                common_properties = process_psf.calculate_psf_properties(mean_image, debug_contours, debug_prefix_mean)
                if common_properties:
                    calculated_angle, common_centroid = common_properties
                    self._logger.info(f"Properties from mean image: Angle={calculated_angle:.2f}, Centroid={common_centroid}")
                    if position in position_angles:
                        common_angle = position_angles[position]
                        angle_source = f"Predefined ({common_angle:.2f} deg)"
                    else:
                        common_angle = calculated_angle
                        angle_source = f"Calculated ({common_angle:.2f} deg)"
                else:
                    self._logger.warning(f"Could not estimate properties from mean image. Using angle=0 and geometric center.")
                    common_centroid = np.array([(shape - 1) / 2.0 for shape in mean_image.shape])
            except Exception as e:
                self._logger.error(f"Error calculating mean image properties: {e}. Using angle=0 and geometric center.")
                common_centroid = np.array([(shape - 1) / 2.0 for shape in mean_image.shape])
        else: # Should not happen if previous check passed, but belts and suspenders
             self._logger.error("Mean image is None, cannot determine common properties.")
             return
             
        self._logger.info(f"Common Angle for Rotation: {common_angle:.2f} (Source: {angle_source})")
        self._logger.info(f"Common Centroid for Rotation Center: {common_centroid}")
        
        # 4. Process each individual frame: Pan based on robust fit, Rotate around common centroid
        for raw_image, header, path, current_wavelength, measured_centroid_y, measured_centroid_x in frame_data: # Renamed for clarity
            self._logger.info(f"Processing frame {os.path.basename(path)} (Wavelength: {current_wavelength} nm)")
            
            panned_image = raw_image # Default to raw_image if panning can't be done
            final_shift_x = 0.0
            final_shift_y = 0.0
            can_pan = True

            # Calculate T_fit_x (target_x_from_fit)
            target_x_from_fit = np.nan
            if x_fit_coeffs is not None:
                target_x_from_fit = x_fit_coeffs[0] * current_wavelength + x_fit_coeffs[1]
            else:
                self._logger.warning(f"  Cannot calculate Target X from fit for {os.path.basename(path)} (X-fit coeffs missing). X-panning component will be zero.")
                # final_shift_x will remain 0.0

            # Calculate T_fit_y (target_y_from_fit)
            target_y_from_fit = np.nan
            if y_fit_coeffs is not None:
                target_y_from_fit = y_fit_coeffs[0] * current_wavelength + y_fit_coeffs[1]
            else:
                self._logger.warning(f"  Cannot calculate Target Y from fit for {os.path.basename(path)} (Y-fit coeffs missing). Y-panning component will be zero.")
                # final_shift_y will remain 0.0
            
            if common_centroid is None:
                self._logger.error(f"  Cannot perform robust panning for {os.path.basename(path)} (Common centroid is None). Using raw image.")
                can_pan = False

            if can_pan:
                # Calculate final_shift_x = T_avg_x - T_fit_x
                # common_centroid is [Y, X]
                if x_fit_coeffs is not None and common_centroid is not None: # Check x_fit_coeffs again for safety
                    final_shift_x = common_centroid[1] - target_x_from_fit
                    self._logger.info(f"  Robust X-Shift Calc: T_avg_x={common_centroid[1]:.2f}, T_fit_x={target_x_from_fit:.2f}, Shift_x={final_shift_x:.2f}")
                else:
                    self._logger.info(f"  Skipping X component of robust shift calculation for {os.path.basename(path)} (missing X-fit or common_centroid).")
                    # final_shift_x remains 0.0

                # Calculate final_shift_y = T_avg_y - T_fit_y
                if y_fit_coeffs is not None and common_centroid is not None: # Check y_fit_coeffs again for safety
                    final_shift_y = common_centroid[0] - target_y_from_fit
                    self._logger.info(f"  Robust Y-Shift Calc: T_avg_y={common_centroid[0]:.2f}, T_fit_y={target_y_from_fit:.2f}, Shift_y={final_shift_y:.2f}")
                else:
                    self._logger.info(f"  Skipping Y component of robust shift calculation for {os.path.basename(path)} (missing Y-fit or common_centroid).")
                    # final_shift_y remains 0.0

                # Apply the combined shift if any shift component is significant
                if abs(final_shift_x) > 1e-6 or abs(final_shift_y) > 1e-6:
                    self._logger.info(f"  Applying combined robust shift: (dy={final_shift_y:.2f}, dx={final_shift_x:.2f})")
                    try:
                        panned_image = shift(raw_image, shift=(final_shift_y, final_shift_x), order=3, mode='constant', cval=0.0)
                    except Exception as e:
                        self._logger.error(f"  Error applying combined shift for {os.path.basename(path)}: {e}. Using raw image for rotation.")
                        panned_image = raw_image # Fallback
                else:
                    self._logger.info(f"  No significant robust shift to apply for {os.path.basename(path)}. Using raw image as panned_image.")
                    panned_image = raw_image
            else: # can_pan is False
                 self._logger.warning(f"  Skipping robust panning for {os.path.basename(path)} due to missing critical components. Using raw image.")
                 panned_image = raw_image # Ensure panned_image is raw_image

            # Pass the panned_image to _processFrame for rotation and cropping
            try:
                processed_image = self._processFrame(panned_image, common_angle, common_centroid, debug_contours, path)
                
                if processed_image is None: 
                    self._logger.warning(f"Skipping save for frame {os.path.basename(path)} due to processing error in _processFrame.")
                    continue

                # Save the processed image
                new_filename = path.replace('.fits', '_proc.fits')
                # Update header
                header['HISTORY'] = f"Processed by SPL_data_processer."
                header['HISTORY'] = f"Applied common rotation angle: {common_angle:.2f} deg around common centroid"
                header['HISTORY'] = f"Rotation angle source: {angle_source}"
                if can_pan and (abs(final_shift_x) > 1e-6 or abs(final_shift_y) > 1e-6):
                     header['HISTORY'] = f"Applied robust pan (T_avg - T_fit): dy={final_shift_y:.2f}, dx={final_shift_x:.2f} px"
                elif not can_pan:
                     header['HISTORY'] = f"Robust panning skipped due to missing data (fit coeffs or common_centroid)."
                else:
                     header['HISTORY'] = f"Robust panning resulted in zero significant shift."

                # Add details about measured vs fitted if helpful for diagnostics (optional)
                # if not np.isnan(measured_centroid_x) and not np.isnan(target_x_from_fit):
                #    header[f'DX_MEAS_VS_FIT'] = (measured_centroid_x - target_x_from_fit, 'Measured X_cen - Fit X_cen for this WL')
                # if not np.isnan(measured_centroid_y) and not np.isnan(target_y_from_fit):
                #    header[f'DY_MEAS_VS_FIT'] = (measured_centroid_y - target_y_from_fit, 'Measured Y_cen - Fit Y_cen for this WL')
                
                pyfits.writeto(new_filename, processed_image, header=header, overwrite=True)
                self._logger.info(f"Saved processed image to {new_filename}")
                
            except Exception as e:
                self._logger.error(f"Error processing/saving frame from {path}: {e}")

        # --- Debug Plots --- (Existing code for Centroid Y/X plots, using originally calculated centroids)
        # We plot the originally measured centroids, not the target ones
        plot_wavelengths = [fd[3] for fd in frame_data] 
        plot_centroid_y = [fd[4] for fd in frame_data]
        plot_centroid_x = [fd[5] for fd in frame_data]
        # ... (Generate Y Plot using plot_wavelengths, plot_centroid_y)
        # ... (Generate X Plot using plot_wavelengths, plot_centroid_x)
        if debug_contours and plot_wavelengths:
             try:
                  plt.figure(figsize=(10, 6))
                  plt.plot(plot_wavelengths, plot_centroid_y, marker='o', linestyle='-', label='Measured Y')
                  # Optionally plot the fitted line
                  if y_fit_coeffs is not None:
                      fit_y = np.polyval(y_fit_coeffs, plot_wavelengths)
                      plt.plot(plot_wavelengths, fit_y, linestyle='--', color='g', label='Weighted Fit Y')
                  plt.xlabel("Wavelength (nm)")
                  plt.ylabel("Centroid Y-coordinate (pixels)")
                  plt.title(f"Position {position:02d}: Centroid Y vs Wavelength")
                  plt.legend()
                  plt.grid(True)
                  plot_filename = os.path.join(folder, f"pos{position:02d}_centroid_Y_vs_wavelength.png")
                  plt.savefig(plot_filename)
                  plt.close()
                  self._logger.info(f"Saved centroid Y debug plot to {plot_filename}")
             except Exception as e:
                  self._logger.warning(f"Could not generate or save centroid Y debug plot for position {position:02d}: {e}")
                  
             try:
                  plt.figure(figsize=(10, 6))
                  plt.plot(plot_wavelengths, plot_centroid_x, marker='x', linestyle='--', color='r', label='Measured X')
                   # Optionally plot the fitted line
                  if x_fit_coeffs is not None:
                      fit_x = np.polyval(x_fit_coeffs, plot_wavelengths)
                      plt.plot(plot_wavelengths, fit_x, linestyle='--', color='r', label='Weighted Fit X')
                  plt.xlabel("Wavelength (nm)")
                  plt.ylabel("Centroid X-coordinate (pixels)")
                  plt.title(f"Position {position:02d}: Centroid X vs Wavelength")
                  plt.legend()
                  plt.grid(True)
                  plot_filename_x = os.path.join(folder, f"pos{position:02d}_centroid_X_vs_wavelength.png")
                  plt.savefig(plot_filename_x)
                  plt.close()
                  self._logger.info(f"Saved centroid X debug plot to {plot_filename_x}")
             except Exception as e:
                  self._logger.warning(f"Could not generate or save centroid X debug plot for position {position:02d}: {e}")

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
