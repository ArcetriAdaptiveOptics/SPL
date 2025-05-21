import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import glob
import re
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.ndimage import maximum_position, median_filter

# Import configuration from parent package
from ..conf import configuration as config

def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """
    2D Gaussian function.
    coords: tuple (x,y) of meshgrid coordinates
    """
    x, y = coords
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude * np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))
    return g.ravel()

class CalibrateExposureTime:
    def __init__(self, tt):
        self.tt = tt
        self.logger = self._setup_logger()
        
        try:
            self.E_base = config.EXPTIME # ms
            self.logger.info(f"Using E_base (config.EXPTIME): {self.E_base} ms")
        except AttributeError:
            self.logger.error("config.EXPTIME not found in SPL configuration. Please define it. Using fallback 250ms.")
            self.E_base = 250 # Fallback if EXPTIME is missing
            
        try:
            self.DESIRED_FLUX_LEVEL = config.DESIRED_FLUX_LEVEL # ADU
            self.logger.info(f"Using DESIRED_FLUX_LEVEL (config.DESIRED_FLUX_LEVEL): {self.DESIRED_FLUX_LEVEL} ADU")
        except AttributeError:
            self.logger.error("config.DESIRED_FLUX_LEVEL not found in SPL configuration. Please define it. Using fallback 800 ADU.")
            self.DESIRED_FLUX_LEVEL = 800 # Fallback if DESIRED_FLUX_LEVEL is missing
                
        self.data_folder = self._get_data_folder()
        if not self.data_folder:
            self.logger.error(f"Data folder for TT {tt} could not be determined. Most operations will fail.")

        self.flux_measurements = []

    def _setup_logger(self):
        import logging
        logger = logging.getLogger(f"SplFluxCalibrator_{self.tt}")
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def _get_data_folder(self):
        """Constructs the path to the measurement data folder."""
        measurement_path = config.MEASUREMENT_ROOT_FOLDER
        folder = os.path.join(measurement_path, self.tt)
        if os.path.isdir(folder):
            return folder
        self.logger.error(f"Data folder not found: {folder}")
        return None

    def _parse_files(self):
        """
        Parses FITS filenames to extract wavelengths and positions,
        grouping files by position. Skips '_proc.fits' files.
        Returns: dict
            {position_idx (int): [(wavelength (int), filepath (str)), ...]}
        """
        if not self.data_folder:
            return {}

        file_pattern = re.compile(r"image_(\d{3})nm_pos(\d{2})\.fits")
        position_data = {}

        self.logger.info(f"Scanning folder: {self.data_folder}")
        for filename in sorted(os.listdir(self.data_folder)):
            if filename.endswith("_proc.fits"):
                continue  # Skip processed files

            match = file_pattern.match(filename)
            if match:
                wavelength_str, position_str = match.groups()
                wavelength = int(wavelength_str)
                position = int(position_str)
                filepath = os.path.join(self.data_folder, filename)

                if position not in position_data:
                    position_data[position] = []
                position_data[position].append((wavelength, filepath))
        
        for pos in position_data:
            position_data[pos].sort() # Sort by wavelength

        if not position_data:
            self.logger.warning("No valid FITS files found (e.g., image_XXXnm_posYY.fits).")
        return position_data

    def _measure_flux_2d_fit(self, image_data, roi_size=30, median_filter_size=3):
        """
        Measures flux by fitting a 2D Gaussian to the brightest spot in an ROI.
        Applies a median filter to the image data before fitting to reduce hot pixels.
        Uses the amplitude of the Gaussian as flux.
        """
        if image_data is None or image_data.ndim != 2:
            self.logger.warning("Invalid image data for 2D fit.")
            return np.nan

        # Apply Median Filter to reduce hot pixels
        filtered_image_data = image_data.copy()
        if median_filter_size and median_filter_size > 1:
            try:
                self.logger.debug(f"Applying median filter with size {median_filter_size}x{median_filter_size}.")
                filtered_image_data = median_filter(image_data, size=median_filter_size)
            except Exception as e:
                self.logger.warning(f"Error applying median filter: {e}. Proceeding with original image data for this frame.")

        # Estimate initial centroid from max value
        h, w = filtered_image_data.shape
        
        try:
            y0_guess, x0_guess = maximum_position(filtered_image_data)
        except Exception as e:
            self.logger.warning(f"Could not find initial max position on filtered image, using center: {e}")
            y0_guess, x0_guess = h // 2, w // 2
            
        # Define ROI around the initial guess
        y_min = max(0, int(y0_guess - roi_size // 2))
        y_max = min(h, int(y0_guess + roi_size // 2))
        x_min = max(0, int(x0_guess - roi_size // 2))
        x_max = min(w, int(x0_guess + roi_size // 2))

        roi = filtered_image_data[y_min:y_max, x_min:x_max]
        
        if roi.size == 0:
            self.logger.warning("ROI is empty, cannot fit.")
            return np.nan

        roi_h, roi_w = roi.shape
        x_roi, y_roi = np.meshgrid(np.arange(roi_w), np.arange(roi_h))

        # Initial parameters for the fit
        min_roi_val = np.min(roi)
        amp_guess = max(0, np.max(roi) - min_roi_val) 
        x0_roi_guess, y0_roi_guess = roi_w / 2.0, roi_h / 2.0
        offset_guess = min_roi_val
        
        sigma_x_guess = max(1.0, roi_w / 4.0)
        sigma_y_guess = max(1.0, roi_h / 4.0)

        initial_guess = [amp_guess, x0_roi_guess, y0_roi_guess, sigma_x_guess, sigma_y_guess, 0, offset_guess]
        bounds = (
            [0, 0, 0, 1e-9, 1e-9, -np.pi/2, -np.inf],
            [np.inf, roi_w, roi_h, roi_w, roi_h, np.pi/2, np.inf]
        )

        try:
            popt, pcov = curve_fit(gaussian_2d, (x_roi, y_roi), roi.ravel(), p0=initial_guess, bounds=bounds, maxfev=10000)
            amplitude = popt[0]
            if amplitude < 1e-9:
                self.logger.warning(f"Unreasonable 2D Gaussian fit parameters: Amplitude={amplitude:.2f} is too low. Defaulting to max pixel in ROI.")
                return np.max(roi) if roi.size > 0 else np.nan
            return amplitude
        except RuntimeError:
            self.logger.warning("2D Gaussian fit failed (RuntimeError). Defaulting to max pixel in ROI.")
            return np.max(roi) if roi.size > 0 else np.nan
        except Exception as e:
            self.logger.warning(f"2D Gaussian fit failed ({e}). Defaulting to max pixel in ROI.")
            return np.max(roi) if roi.size > 0 else np.nan

    def process_files(self):
        """Processes all found files to measure flux."""
        position_files = self._parse_files()
        if not position_files:
            self.logger.error("No files to process.")
            return

        for position_idx, files in position_files.items():
            self.logger.info(f"--- Processing Position {position_idx:02d} ---")
            for wavelength, filepath in files:
                self.logger.info(f"  Loading image: {os.path.basename(filepath)} (Wavelength: {wavelength} nm)")
                actual_exptime_for_file = self.E_base
                flux_from_fit = np.nan
                saturated_flag = False

                try:
                    with fits.open(filepath) as hdul:
                        image_data = hdul[0].data
                        header = hdul[0].header
                        
                        file_exptime = header.get('EXPTIME')
                        if file_exptime is not None:
                            try:
                                actual_exptime_for_file = float(file_exptime)
                                self.logger.debug(f"    Read EXPTIME {actual_exptime_for_file:.2f} ms from FITS header.")
                            except ValueError:
                                self.logger.warning(f"    Could not parse EXPTIME '{file_exptime}' from FITS header. Using E_base {self.E_base:.2f} ms.")
                        else:
                            self.logger.debug(f"    EXPTIME not found in FITS header. Using E_base {self.E_base:.2f} ms.")

                        if image_data is None:
                            self.logger.warning(f"    No data in FITS file {os.path.basename(filepath)}. Skipping.")
                        else:
                            image_data = image_data.astype(np.float32)
                            flux_from_fit = self._measure_flux_2d_fit(image_data)
                            
                            if flux_from_fit is not None and not np.isnan(flux_from_fit):
                                if flux_from_fit > 1024:
                                    self.logger.warning(f"    Pos {position_idx}, WL {wavelength}: Raw flux {flux_from_fit:.2f} > 1024. Marking as saturated.")
                                    saturated_flag = True
                                self.logger.info(f"    Measured raw flux (fit amplitude): {flux_from_fit:.2f}")
                            else:
                                self.logger.warning(f"    Flux measurement failed or resulted in NaN for {os.path.basename(filepath)}.")

                except Exception as e:
                    self.logger.error(f"    Error reading or processing FITS file {os.path.basename(filepath)}: {e}")

                flux_to_store_for_plots = 1024.0 if saturated_flag and (flux_from_fit is not None and flux_from_fit > 1024) else flux_from_fit

                self.flux_measurements.append({
                    'position': position_idx,
                    'wavelength': wavelength,
                    'flux': flux_to_store_for_plots,
                    'measured_flux_raw': flux_from_fit,
                    'actual_exptime': actual_exptime_for_file,
                    'saturated_original': saturated_flag,
                    'file': os.path.basename(filepath)
                })
        self.logger.info("Finished processing all files.")

    def calculate_exposure_correction_factors(self):
        """
        Calculates exposure correction factors for each measurement.
        The factor is E_desired / E_base, where E_desired is the exposure
        time needed to reach self.DESIRED_FLUX_LEVEL.
        """
        if not self.flux_measurements:
            self.logger.info("No flux measurements available to calculate correction factors.")
            return

        self.logger.info(f"Calculating exposure correction factors based on DESIRED_FLUX_LEVEL={self.DESIRED_FLUX_LEVEL} ADU and E_base={self.E_base} ms.")
        
        if np.isnan(self.DESIRED_FLUX_LEVEL) or np.isnan(self.E_base) or self.E_base <= 0:
            self.logger.error("DESIRED_FLUX_LEVEL or E_base is invalid. Cannot calculate correction factors.")
            for fm in self.flux_measurements:
                fm['exposure_correction_factor'] = np.nan
            return

        for fm in self.flux_measurements:
            raw_flux = fm.get('measured_flux_raw')
            actual_et = fm.get('actual_exptime')
            is_saturated = fm.get('saturated_original', False)
            
            correction_factor = np.nan

            if raw_flux is not None and not np.isnan(raw_flux) and \
               actual_et is not None and not np.isnan(actual_et) and actual_et > 1e-9:

                if abs(raw_flux) > 1e-9:
                    E_desired = actual_et * (self.DESIRED_FLUX_LEVEL / raw_flux)
                    correction_factor = E_desired / self.E_base
                    
                    log_msg_extra = ""
                    if is_saturated:
                        log_msg_extra = f" (Original flux {raw_flux:.2f} was saturated)"

                    self.logger.debug(
                        f"  Pos {fm['position']}, WL {fm['wavelength']}: "
                        f"RawFlux={raw_flux:.2f}, ActualET={actual_et:.2f}ms. "
                        f"E_desired={E_desired:.2f}ms, CorrFactor={correction_factor:.3f}.{log_msg_extra}"
                    )
                else:
                    self.logger.debug(
                        f"  Pos {fm['position']}, WL {fm['wavelength']}: "
                        f"RawFlux={raw_flux:.2f} is zero or too small. CorrFactor set to NaN."
                    )
            else:
                self.logger.debug(
                    f"  Pos {fm['position']}, WL {fm['wavelength']}: "
                    f"Missing raw flux ({raw_flux}) or actual exposure time ({actual_et}). CorrFactor set to NaN."
                )
            
            fm['exposure_correction_factor'] = correction_factor
        
        self.logger.info("Finished calculating exposure correction factors.")

    def _calculate_mean_correction_factors_per_wavelength(self):
        """
        Calculates the mean exposure correction factor for each wavelength.
        """
        if not self.flux_measurements:
            self.logger.info("No flux measurements available to calculate mean correction factors.")
            self.mean_wavelength_factors = []
            return

        self.logger.info("Calculating mean exposure correction factors per wavelength.")
        self.mean_wavelength_factors = []
        
        if np.isnan(self.DESIRED_FLUX_LEVEL) or np.isnan(self.E_base) or self.E_base <=0:
            self.logger.error("DESIRED_FLUX_LEVEL or E_base is invalid. Cannot calculate mean correction factors.")
            unique_wavelengths_for_nan = sorted(list(set(fm['wavelength'] for fm in self.flux_measurements)))
            for wl_nan in unique_wavelengths_for_nan:
                 self.mean_wavelength_factors.append({
                    'WAVELENGTH_NM': wl_nan,
                    'MEAN_EXPOSURE_CORR_FACTOR': np.nan
                })
            return

        unique_wavelengths = sorted(list(set(fm['wavelength'] for fm in self.flux_measurements)))
        
        for wl in unique_wavelengths:
            wl_raw_fluxes = []
            wl_actual_exptimes = []

            for fm in self.flux_measurements:
                if fm['wavelength'] == wl:
                    raw_flux = fm.get('measured_flux_raw')
                    actual_et = fm.get('actual_exptime')
                    if raw_flux is not None and not np.isnan(raw_flux) and \
                       actual_et is not None and not np.isnan(actual_et):
                        wl_raw_fluxes.append(raw_flux)
                        wl_actual_exptimes.append(actual_et)
            
            mean_factor_for_wl = np.nan
            avg_raw_flux_wl = np.nan
            avg_actual_exptime_wl = np.nan

            if wl_raw_fluxes and wl_actual_exptimes:
                avg_raw_flux_wl = np.nanmean(wl_raw_fluxes)
                avg_actual_exptime_wl = np.nanmean(wl_actual_exptimes)

                if not np.isnan(avg_raw_flux_wl) and abs(avg_raw_flux_wl) > 1e-9 and \
                   not np.isnan(avg_actual_exptime_wl) and avg_actual_exptime_wl > 1e-9:
                    
                    E_desired_for_avg = avg_actual_exptime_wl * (self.DESIRED_FLUX_LEVEL / avg_raw_flux_wl)
                    mean_factor_for_wl = E_desired_for_avg / self.E_base
                    
                    self.logger.debug(
                        f"  Wavelength {wl} nm: AvgRawFlux={avg_raw_flux_wl:.2f}, AvgActualET={avg_actual_exptime_wl:.2f}ms. "
                        f"E_desired_for_avg={E_desired_for_avg:.2f}ms, MeanCorrFactor={mean_factor_for_wl:.3f}"
                    )
                else:
                    self.logger.warning(
                        f"  Wavelength {wl} nm: AvgRawFlux ({avg_raw_flux_wl:.2f}) or AvgActualET ({avg_actual_exptime_wl:.2f}ms) is invalid or zero. Mean factor set to NaN."
                    )
            else:
                self.logger.warning(f"  Wavelength {wl} nm: No valid raw fluxes or actual exposure times to average. Mean factor set to NaN.")
            
            self.mean_wavelength_factors.append({
                'WAVELENGTH_NM': wl,
                'MEAN_EXPOSURE_CORR_FACTOR': mean_factor_for_wl
            })
        self.logger.info("Finished calculating mean exposure correction factors per wavelength.")

    def plot_fluxes(self):
        """Plots measured fluxes as a function of wavelength for each position."""
        if not self.flux_measurements:
            self.logger.info("No flux measurements to plot.")
            return None

        plt.figure(figsize=(12, 7))
        
        positions = sorted(list(set(fm['position'] for fm in self.flux_measurements)))

        for pos_idx in positions:
            pos_data = [fm for fm in self.flux_measurements if fm['position'] == pos_idx]
            if not pos_data:
                continue
            
            wavelengths = np.array([d['wavelength'] for d in pos_data])
            fluxes = np.array([d['flux'] for d in pos_data])
            
            sort_indices = np.argsort(wavelengths)
            wavelengths_sorted = wavelengths[sort_indices]
            fluxes_sorted = fluxes[sort_indices]

            plt.plot(wavelengths_sorted, fluxes_sorted, marker='o', linestyle='-', label=f'Position {pos_idx:02d}')

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Measured Flux (Fit Amplitude)")
        plt.title(f"Flux vs. Wavelength for TT: {self.tt}")
        plt.legend()
        plt.grid(True)
        
        tt_folder_name = os.path.basename(os.path.normpath(self.data_folder))
        plot_filename = os.path.join(self.data_folder, f"{tt_folder_name}_flux_vs_wavelength.png")
        try:
            plt.savefig(plot_filename)
            self.logger.info(f"Flux plot saved to: {plot_filename}")
            plt.close()
            return plot_filename
        except Exception as e:
            self.logger.error(f"Error saving plot: {e}")
            plt.close()
            return None

    def plot_mean_correction_factors(self):
        """Plots mean exposure correction factors as a function of wavelength."""
        if not hasattr(self, 'mean_wavelength_factors') or not self.mean_wavelength_factors:
            self.logger.info("No mean wavelength factors to plot.")
            return None

        plt.figure(figsize=(12, 7))
        
        wavelengths = np.array([mf['WAVELENGTH_NM'] for mf in self.mean_wavelength_factors])
        mean_factors = np.array([mf['MEAN_EXPOSURE_CORR_FACTOR'] for mf in self.mean_wavelength_factors])
        
        valid_indices = ~np.isnan(mean_factors)
        wavelengths_plot = wavelengths[valid_indices]
        mean_factors_plot = mean_factors[valid_indices]

        if len(wavelengths_plot) == 0:
            self.logger.info("No valid (non-NaN) mean correction factors to plot.")
            plt.close()
            return None

        sort_indices = np.argsort(wavelengths_plot)
        wavelengths_sorted = wavelengths_plot[sort_indices]
        mean_factors_sorted = mean_factors_plot[sort_indices]

        plt.plot(wavelengths_sorted, mean_factors_sorted, marker='o', linestyle='-', color='green')

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Mean Exposure Correction Factor")
        plt.title(f"Mean Exposure Correction Factor vs. Wavelength for TT: {self.tt}")
        plt.grid(True)
        plt.ylim(bottom=0)

        tt_folder_name = os.path.basename(os.path.normpath(self.data_folder))
        plot_filename = os.path.join(self.data_folder, f"{tt_folder_name}_mean_correction_factors.png")
        try:
            plt.savefig(plot_filename)
            self.logger.info(f"Mean correction factor plot saved to: {plot_filename}")
            plt.close()
            return plot_filename
        except Exception as e:
            self.logger.error(f"Error saving mean correction factor plot: {e}")
            plt.close()
            return None

    def save_results_to_fits(self):
        """Saves the flux measurements and correction factors to a FITS file."""
        if not self.flux_measurements:
            self.logger.info("No flux measurements to save.")
            return None

        for fm in self.flux_measurements:
            if 'exposure_correction_factor' not in fm:
                fm['exposure_correction_factor'] = np.nan

        data_for_table1 = {
            'POSITION': [fm['position'] for fm in self.flux_measurements],
            'WAVELENGTH_NM': [fm['wavelength'] for fm in self.flux_measurements],
            'MEASURED_FLUX': [fm['flux'] for fm in self.flux_measurements],
            'EXPOSURE_CORR_FACTOR': [fm['exposure_correction_factor'] for fm in self.flux_measurements],
            'SOURCE_FILE': [fm['file'] for fm in self.flux_measurements]
        }
        table1 = Table(data_for_table1)
        table1.sort(['POSITION', 'WAVELENGTH_NM'])
        hdu1 = fits.BinTableHDU(data=table1, name='PER_POSITION_CORRECTIONS')

        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['TTPROC'] = (self.tt, 'Tracking number of this flux calibration')
        primary_hdu.header['E_BASE'] = (self.E_base, 'Base exposure time from config (ms)')
        primary_hdu.header['FLUXLVL'] = (self.DESIRED_FLUX_LEVEL, 'Desired flux level (ADU)')
        primary_hdu.header['COMMENT'] = "HDU1 (PER_POSITION_CORRECTIONS): Detailed flux & per-position correction factors."
        primary_hdu.header['COMMENT'] = "  Flux measured by 2D Gaussian fit amplitude (MEASURED_FLUX is capped at 1024)."
        primary_hdu.header['COMMENT'] = "  Correction factor calculated based on uncapped 'measured_flux_raw',"
        primary_hdu.header['COMMENT'] = "  'actual_exptime' (from FITS header or E_base), DESIRED_FLUX_LEVEL, and E_base."
        primary_hdu.header['COMMENT'] = "  EXPOSURE_CORR_FACTOR = (actual_exptime * (DESIRED_FLUX_LEVEL / measured_flux_raw)) / E_base."
        
        hdulist = fits.HDUList([primary_hdu, hdu1])

        if hasattr(self, 'mean_wavelength_factors') and self.mean_wavelength_factors:
            data_for_table2 = {
                'WAVELENGTH_NM': [mf['WAVELENGTH_NM'] for mf in self.mean_wavelength_factors],
                'MEAN_EXPOSURE_CORR_FACTOR': [mf['MEAN_EXPOSURE_CORR_FACTOR'] for mf in self.mean_wavelength_factors]
            }
            table2 = Table(data_for_table2)
            table2.sort(['WAVELENGTH_NM'])
            hdu2 = fits.BinTableHDU(data=table2, name='MEAN_WAVELENGTH_CORRECTIONS')
            hdulist.append(hdu2)
            primary_hdu.header['COMMENT'] = "HDU2 (MEAN_WAVELENGTH_CORRECTIONS): Mean correction factor per wavelength."
            primary_hdu.header['COMMENT'] = "  Calculated based on average raw flux and average actual exposure per wavelength,"
            primary_hdu.header['COMMENT'] = "  DESIRED_FLUX_LEVEL, and E_base."
        else:
            self.logger.info("No mean wavelength correction factors to save in HDU2.")

        tt_folder_name = os.path.basename(os.path.normpath(self.data_folder))
        output_fits_filename = os.path.join(self.data_folder, f"{tt_folder_name}_flux_calibration.fits")
        
        try:
            hdulist.writeto(output_fits_filename, overwrite=True)
            self.logger.info(f"Flux calibration data saved to: {output_fits_filename}")
            return output_fits_filename
        except Exception as e:
            self.logger.error(f"Error saving FITS file: {e}")
            return None

    def run_calibration(self):
        """Runs the full flux calibration process."""
        self.process_files()
        self.plot_fluxes()
        self.calculate_exposure_correction_factors()
        self._calculate_mean_correction_factors_per_wavelength()
        self.plot_mean_correction_factors()
        output_fits_path = self.save_results_to_fits()
        return output_fits_path 