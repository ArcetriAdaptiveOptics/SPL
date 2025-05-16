'''
Authors
  - G. Pariani, R.Briguglio: written in 2016
  - C. Selmi: ported to Python in 2020
  - A. Puglisi and C. Selmi: python code debugging in 2021
  - M. Xompero and N. Azzaroli: modified SplAnalyzer from **5 to **.5
  - M. Bonaglia: added code to manage more than 1 spot per frame (using pos idx in fits filenames) in 03/2025
  - M. Bonaglia: added code to convert piston values from nanometers to meters in 03/2025
  - M. Bonaglia: added code to interpolate matrices to match Qt shape in 03/2025
'''
import os
import numpy as np
import glob
import logging
from astropy.io import fits as pyfits
from photutils import centroids
#import matplotlib.pyplot as plt
from tqdm import tqdm
from spl.ground import smooth_function as sf
from spl.conf import configuration as config
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from astropy.io import fits
import re
from scipy.interpolate import interp1d
# Import needed for template matching
from skimage.feature import match_template
import argparse # Add argparse import
# from scipy import signal # Keep commented for now


class SplAnalyzer():
    '''
    Class used to analyze images, whit SPL camera, at different wavelengths.

    HOW TO USE IT::
        from spl import SPL_data_analyzer import SplAnalyzer
        an = SplAnalyzer()
        piston, piston_smooth = an.analyzer(tt)
    '''

    def __init__(self):
        """The constructor """
        self._logger = logging.getLogger('SPL_AN:')
        self.tn_fringes = config.TNFRINGES
        self._Qm = None
        self._QmSmooth = None
        self._matrix = None
        self._matrixSmooth = None

    @staticmethod
    def _storageFolder():
        """ Creates the path where to save measurement data"""
        measurement_path = config.MEASUREMENT_ROOT_FOLDER
        return measurement_path
    
    @staticmethod
    def _storageFringesFolder():
        ''' Path for fringes frames to use in analysis comparation'''
        fringes_path = os.path.join(os.path.dirname(__file__), 'Fringes')
        return fringes_path
    
    def analyzer(self, tt, use_processed=False, matching_method='original'):
        '''
        Analyze measurement data and compare it with synthetic data.

        Parameters
        ----------
        tt: string
            tracking number of the measurement data
        use_processed: bool
            whether to use processed FITS files
        matching_method: str, optional
            Method for template comparison ('original', 'cross_correlation').
            Defaults to 'original'.
        Returns
        -------
        pistons: list of tuples
            list of (piston, piston_smooth) for each position
        '''
        dove = os.path.join(self._storageFolder(), tt)

        self._logger.info('Analysis of tt = %s using matching method: %s', tt, matching_method)

        lambda_path = os.path.join(dove, 'lambda_vector.fits')
        hduList = pyfits.open(lambda_path)
        lambda_vector = hduList[0].data  # Assign lambda_vector here

        # Extract wavelengths and positions
        wavelengths, positions = self.parseFitsFilenames(dove)
        #print(wavelengths)
        #print(positions)

        pistons = []
        
        # For each position process the images
        for position in tqdm(positions, desc="Analyzing Positions", unit="pos"):
            cube, cube_normalized = self._readMeasurement(position, tt, use_processed)
            if cube is None: # Handle case where no files were found/read
                self._logger.warning(f"No valid data cube read for position {position}. Skipping analysis for this position.")
                pistons.append((np.nan, np.nan)) # Or some other indicator
                continue
            
            # Compute matrix and smoothed matrix
            matrix, matrix_smooth = self.matrix_calc(lambda_vector, cube, cube_normalized)

            # Compare with synthetic data to get piston and piston_smooth
            #print('*** Position = ', position)
            piston, piston_smooth = self._templateComparison(matrix, matrix_smooth, lambda_vector, tt, position, matching_method)

            # Save the piston result
            self._savePistonResult(tt, piston, piston_smooth, position)

            # Store the piston results
            pistons.append((piston, piston_smooth))
        
        return pistons
 
    def parseFitsFilenames(self, data_folder):
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

    def _readMeasurement(self, position, tt, use_processed):
        """ Read images from a specific tracking number and return the cube.

        Args:
            position (int): Position index (e.g., 00, 01, etc.)
            tt (str): Tracking number.
            use_processed (bool): Whether to use processed FITS files

        Returns:
            tuple: 
                - cube (numpy.ndarray): [pixels, pixels, n_frames]
                - cube_normalized (numpy.ndarray): Normalized images, [pixels, pixels, n_frames]
        """
        dove = os.path.join(self._storageFolder(), tt)
        
        # Choose the file pattern based on the use_processed flag
        if use_processed:
            file_pattern = f'image_*nm_pos{position:02d}_proc.fits'
            self._logger.info(f"Reading PROCESSED files for position {position:02d} with pattern: {file_pattern}")
        else:
            file_pattern = f'image_*nm_pos{position:02d}.fits'
            self._logger.info(f"Reading RAW files for position {position:02d} with pattern: {file_pattern}")
            
        # Get sorted list of FITS files matching the specific position and pattern
        path_list = sorted(glob.iglob(os.path.join(dove, file_pattern)))
        #print(path_list)

        if not path_list:
            self._logger.error(f"Error: No FITS files found for position {position:02d} in {dove} using pattern '{file_pattern}'")
            return None, None  # Handle case with no images

        cube = []
        cube_normalized = []

        for path in tqdm(path_list, desc=f"Reading Pos {position:02d} Files", unit="file", leave=False):
            try:
                with pyfits.open(path) as hduList:
                    image = hduList[0].data
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue  # Skip corrupted files

            if image is None:
                print(f"Warning: No data in {path}")
                continue  # Skip empty images

            # Clip negative values to zero after reading
            image = np.clip(image, 0, None)

            image_sum = np.sum(image)
            image_normalized = image / image_sum if image_sum != 0 else image  # Prevent division by zero
            
            cube.append(image)
            cube_normalized.append(image_normalized)

        if not cube:
            print(f"Error: No valid images found for position {position:02d}!")
            return None, None  # Handle case where all files were skipped

        # Stack images along the third dimension
        return np.dstack(cube), np.dstack(cube_normalized)

    def _saveMatrix(self, matrix, tt, wavelength, position):
        """ Save matrix to the file system.
        This method might become obsolete if _templateComparison handles saving Qm directly.

        Args:
            matrix (numpy.ndarray): The matrix to save.
            tt (str): Tracking number.
            wavelength (int): Wavelength.
            position (int): Position.
        """
        destination_file_path = self._storageFolder()
        fits_file_name = os.path.join(destination_file_path, tt, f'fringe_result_pos{position:02d}.fits')

    def matrix_calc(self, lambda_vector, cube, cube_normalized):
        img = np.sum(cube_normalized, 2)
        pick = self._newThr(img)
        
        # Ensure pick values are within valid bounds
        height, width = img.shape
        pick[0] = max(0, min(pick[0], height-1))
        pick[1] = max(0, min(pick[1], height-1))
        pick[2] = max(0, min(pick[2], width-1))
        pick[3] = max(0, min(pick[3], width-1))
        
        # Calculate matrix dimensions
        matrix_height = pick[3] - pick[2] + 1
        matrix_width = lambda_vector.shape[0]
        
        # Initialize matrices with correct dimensions
        matrix = np.zeros((matrix_height, matrix_width))
        matrix_smooth = np.zeros((matrix_height, matrix_width))
        crop_frame_cube = None
        
        for i in range(lambda_vector.shape[0]):
            frame = cube[:, :, i]
            crop_frame = frame[pick[0]:pick[1], pick[2]:pick[3] + 1]

            if crop_frame_cube is None:
                crop_frame_cube = crop_frame
            else:
                crop_frame_cube = np.dstack((crop_frame_cube, crop_frame))

            y = np.sum(crop_frame, 0)
            area = np.sum(y[:])
            y_norm = y / area if area != 0 else y
            
            # Ensure y_norm has the correct length
            if len(y_norm) != matrix_height:
                self._logger.warning(f"Length mismatch in y_norm: {len(y_norm)} vs expected {matrix_height}")
                # Pad or truncate y_norm to match matrix_height
                if len(y_norm) < matrix_height:
                    y_norm = np.pad(y_norm, (0, matrix_height - len(y_norm)), mode='constant')
                else:
                    y_norm = y_norm[:matrix_height]
            
            matrix[:, i] = y_norm
            
            # Smooth the normalized data
            w = sf.smooth(y_norm, 3)
            #w=y_norm

            # Ensure smoothed data has correct length
            if len(w) != matrix_height:
                self._logger.warning(f"Length mismatch in smoothed data: {len(w)} vs expected {matrix_height}")
                if len(w) < matrix_height:
                    w = np.pad(w, (0, matrix_height - len(w)), mode='constant')
                else:
                    w = w[:matrix_height]
            
            matrix_smooth[:, i] = w

        # Replace NaN values with 0
        matrix = np.nan_to_num(matrix, nan=0.0)
        matrix_smooth = np.nan_to_num(matrix_smooth, nan=0.0)
        
        self._matrix = matrix
        self._matrixSmooth = matrix_smooth
        return matrix, matrix_smooth

    def get_matrix(self, tn):
        '''
        Paramenters
        -----------
        tn: string
            tracking number from which to read matrix

        Returns
        -------
        matrix: numpy array
            matrix of fringes
        '''
        destination_file_path = self._storageFolder()
        fits_file_name = os.path.join(destination_file_path, tn, 'fringe_result.fits')
        
        hduList = pyfits.open(fits_file_name)
        matrix = hduList[0].data
        return matrix    
    
    def _newThr(self, img):
        ''' Calculate the peak position of the image '''
        #histogram thr to be discussed
#        counts, bin_edges = np.histogram(img)
#        edges = (bin_edges[2:] + bin_edges[1:len(bin_edges)-1]) / 2
#        thr = 5 * edges[np.where(counts == max(counts))]
#        idx = np.where(img < thr)
#        img[idx] = 0
        # cy, cx = scin.measurements.center_of_mass(img)
        cx, cy = centroids.centroid_2dg(img)
        #plt.imshow(img)
        #plt.show()
        #print('Baricenter = ', cx, cy)
        baricenterCoord = [int(round(cy)), int(round(cx))]
        pick = [baricenterCoord[0]-25, baricenterCoord[0]+25,
                baricenterCoord[1]-50, baricenterCoord[1]+50]
        return pick

    def _templateComparison(self, matrix, matrix_smooth, lambda_vector, tt, position, matching_method):
        '''
        Compare the matrix obtained from the measurements with
        the one recreated with the synthetic data in tn_fringes.
        Parameters
        ----------
        matrix: [pixels, lambda]
                measured matrix
        matrix_smooth: [pixels, lambda]
                        measured matrix smooth
        lambda_vector: numpy array
                        vector of wavelengths (between 400/700 nm)
        tt: string
            tracking number (for saving Qm)
        position: int
            position index (for saving Qm)
        matching_method: str
            Method for template comparison ('original', 'cross_correlation')
        Returns
        -------
        piston: int
                piston value
        piston_smooth: int
                        piston smooth value (may be same as piston for new methods)
        '''
        self._logger.debug('Template Comparison with data in tn_fringes = %s using method = %s', 
                           self.tn_fringes, matching_method)
        dove = os.path.join(self._storageFringesFolder(),
                            self.tn_fringes)
        delta, lambda_synth_from_data = self._readDeltaAndLambdaFromFringesFolder(dove)
        lambda_synth = self._myLambaSynth(lambda_synth_from_data)

        # Find common wavelengths between measured and synthetic data
        common_wavelengths = np.intersect1d(lambda_vector, lambda_synth)
        if len(common_wavelengths) == 0:
            raise ValueError("No common wavelengths found between measured and synthetic data")

        # Get indices for common wavelengths
        idx_measured = np.isin(lambda_vector, common_wavelengths)
        idx_synthetic = np.isin(lambda_synth, common_wavelengths)

        # Select only common wavelengths
        Qm = matrix[:, idx_measured] - np.mean(matrix[:, idx_measured])
        Qm_smooth = matrix_smooth[:, idx_measured] - np.mean(matrix_smooth[:, idx_measured])
        self._QmSmooth = Qm_smooth
        # plt.plot(Qm); plt.show() # Original line

        # Create synthetic fringe matrix with common wavelengths
        F = []
        for i in tqdm(range(1, delta.shape[0]), desc="Loading Synthetic Fringes", unit="fringe", leave=False):
            file_name = os.path.join(dove, 'Fringe_%05d.fits' %i)
            hduList = pyfits.open(file_name)
            fringe = hduList[0].data
            fringe_selected = fringe[:, idx_synthetic]
            F.append(fringe_selected)
        F = np.dstack(F)
        Qt = F - np.mean(F)
        
        # --- Normalize Qt slices to [0, 1] if not already ---
        for i in range(Qt.shape[2]):
            slice_qt = Qt[:, :, i]
            min_val = np.nanmin(slice_qt)
            max_val = np.nanmax(slice_qt)
            
            needs_norm = True
            if np.isclose(min_val, max_val): # Constant slice
                if min_val >= -1e-9 and min_val <= 1 + 1e-9: # Check if constant is already in [0,1]
                    needs_norm = False
            elif (min_val >= -1e-9 and max_val <= 1 + 1e-9 and 
                  np.isclose(min_val, 0) and np.isclose(max_val, 1)): # Already spans [0,1]
                needs_norm = False
            elif min_val >= -1e-9 and max_val <= 1 + 1e-9: # Already within [0,1] but not spanning it
                 # Decide if these should also be scaled to span 0-1. For now, let's scale them.
                 # To leave them as is if within [0,1], uncomment below and comment out scaling:
                 # needs_norm = False 
                 pass


            if needs_norm:
                if np.isclose(min_val, max_val):
                    Qt[:, :, i] = np.full_like(slice_qt, 0.0)
                else:
                    # Ensure min_val is not greater than max_val after potential nanmin/nanmax issues with all-nan slices
                    if min_val > max_val: # Should not happen if not all NaNs
                        Qt[:, :, i] = np.full_like(slice_qt, np.nan) # Or 0.0
                    else:
                        Qt[:, :, i] = (slice_qt - min_val) / (max_val - min_val)
        # --- End Qt normalization ---
        
        # Check and match matrix shapes
        if Qt.shape[0] != Qm.shape[0]:
            # Create interpolation function for each wavelength
            x_original = np.linspace(0, 1, Qm.shape[0])
            x_new = np.linspace(0, 1, Qt.shape[0])
            
            # Interpolate Qm
            Qm_interp = np.zeros((Qt.shape[0], Qm.shape[1]))
            for i in range(Qm.shape[1]):
                f = interp1d(x_original, Qm[:, i], kind='linear')
                Qm_interp[:, i] = f(x_new)
            
            # Interpolate Qm_smooth
            x_original_smooth = np.linspace(0, 1, Qm_smooth.shape[0])
            Qm_smooth_interp = np.zeros((Qt.shape[0], Qm_smooth.shape[1]))
            for i in range(Qm_smooth.shape[1]):
                f = interp1d(x_original_smooth, Qm_smooth[:, i], kind='linear')
                Qm_smooth_interp[:, i] = f(x_new)
            
            Qm = Qm_interp
            Qm_smooth = Qm_smooth_interp

        # --- Normalize Qm to [0, 1] if not already ---
        min_qm = np.nanmin(Qm)
        max_qm = np.nanmax(Qm)
        needs_norm_qm = True
        if np.isclose(min_qm, max_qm):
            if min_qm >= -1e-9 and min_qm <= 1 + 1e-9: needs_norm_qm = False
        elif (min_qm >= -1e-9 and max_qm <= 1 + 1e-9 and
              np.isclose(min_qm, 0) and np.isclose(max_qm, 1)):
            needs_norm_qm = False
        elif min_qm >= -1e-9 and max_qm <= 1 + 1e-9: # Already within [0,1] but not spanning it
            # needs_norm_qm = False # Uncomment to leave as is if within [0,1]
            pass


        if needs_norm_qm:
            if np.isclose(min_qm, max_qm):
                Qm = np.full_like(Qm, 0.0)
            else:
                if min_qm > max_qm:
                    Qm = np.full_like(Qm, np.nan) # Or 0.0
                else:
                    Qm = (Qm - min_qm) / (max_qm - min_qm)
        # --- End Qm normalization ---

        # --- Normalize Qm_smooth to [0, 1] if not already ---
        min_qms = np.nanmin(Qm_smooth)
        max_qms = np.nanmax(Qm_smooth)
        needs_norm_qms = True
        if np.isclose(min_qms, max_qms):
            if min_qms >= -1e-9 and min_qms <= 1 + 1e-9: needs_norm_qms = False
        elif (min_qms >= -1e-9 and max_qms <= 1 + 1e-9 and
              np.isclose(min_qms, 0) and np.isclose(max_qms, 1)):
            needs_norm_qms = False
        elif min_qms >= -1e-9 and max_qms <= 1 + 1e-9: # Already within [0,1] but not spanning it
            # needs_norm_qms = False # Uncomment to leave as is if within [0,1]
            pass
            
        if needs_norm_qms:
            if np.isclose(min_qms, max_qms):
                Qm_smooth = np.full_like(Qm_smooth, 0.0)
            else:
                if min_qms > max_qms:
                    Qm_smooth = np.full_like(Qm_smooth, np.nan) # Or 0.0
                else:
                    Qm_smooth = (Qm_smooth - min_qms) / (max_qms - min_qms)
        # --- End Qm_smooth normalization ---

        self._Qm = Qm
        self._Qt = Qt

        # --- Plot and Save Qm matrix (imshow) --- 
        try:
            plt.figure(figsize=(8, 6))
            # Use imshow to display the 2D Qm matrix
            im = plt.imshow(Qm, aspect='auto', cmap='viridis', 
                            interpolation='nearest', origin='lower',
                            extent=[common_wavelengths.min(), common_wavelengths.max(), 0, Qm.shape[0]])
            plt.colorbar(im, label='Mean-Subtracted Normalized Intensity')
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Pixel Row Index in Crop")
            plt.title(f"Position {position:02d}: Measured Fringe Matrix (Qm)")
            
            qm_plot_filename = os.path.join(self._storageFolder(), tt, f'fringe_2D_pos{position:02d}.png')
            plt.savefig(qm_plot_filename)
            plt.close() # Close plot to prevent display
            self._logger.info(f"Saved Qm plot for position {position:02d} to {qm_plot_filename}")
        except Exception as e:
             self._logger.error(f"Could not plot or save Qm matrix for position {position:02d}: {e}")
        # --- End Plot and Save Qm (imshow) ---

        # --- Plot and Save Qm matrix (1D Profiles) --- 
        try:
            plt.figure(figsize=(10, 6))
            num_pixels, num_waves = Qm.shape
            # Create a sequence of colors from a colormap
            colors = plt.cm.viridis(np.linspace(0, 1, num_waves))
            
            for j in range(num_waves):
                plt.plot(range(num_pixels), Qm[:, j], color=colors[j], linewidth=0.8) # Plot each column (wavelength)
                
            plt.xlabel("Pixel Row Index in Crop")
            plt.ylabel("Mean-Subtracted Normalized Intensity")
            plt.title(f"Position {position:02d}: Qm Profiles per Wavelength")
            plt.grid(True, linestyle=':', alpha=0.6)
            # Optional: Add a color bar indicating wavelength? Maybe too complex without legend.
            
            qm_profiles_filename = os.path.join(self._storageFolder(), tt, f'fringe_profiles_pos{position:02d}.png')
            plt.savefig(qm_profiles_filename)
            plt.close() # Close plot to prevent display
            self._logger.info(f"Saved Qm profiles plot for position {position:02d} to {qm_profiles_filename}")
        except Exception as e:
             self._logger.error(f"Could not plot or save Qm profiles for position {position:02d}: {e}")
        # --- End Plot and Save Qm (1D Profiles) ---

        # --- Save Qm matrix data (now saving to fringe_result_pos...fits) --- 
        try:
            fringe_result_filename = os.path.join(self._storageFolder(), tt, f'fringe_result_pos{position:02d}.fits')
            
            primary_hdu = pyfits.PrimaryHDU(data=Qm) # Qm is the processed matrix
            header = primary_hdu.header

            num_common_waves = len(common_wavelengths)
            header['NCOMWAVE'] = (num_common_waves, 'Number of common wavelengths')
            if num_common_waves > 0:
                header['MINCOMWV'] = (float(np.min(common_wavelengths)), 'Min common wavelength (nm)')
                header['MAXCOMWV'] = (float(np.max(common_wavelengths)), 'Max common wavelength (nm)')
                header['CTYPE1'] = ('PIXEL_ROW', 'Label for axis 1 (rows)') # Qm rows are pixel rows
                header['CTYPE2'] = ('WAVELENGTH', 'Label for axis 2 (cols)') # Qm cols are wavelengths
                header['CUNIT2'] = ('nm', 'Unit for axis 2')

                # Calculate and store step if wavelengths are evenly spaced
                if num_common_waves > 1:
                    diffs = np.diff(common_wavelengths)
                    step = diffs[0]
                    if np.allclose(diffs, step):
                        header['STEPCOMW'] = (float(step), 'Step of common wavelengths (nm)')
                    else:
                        header['STEPCOMW'] = ('Non-uniform', 'Step of common wavelengths (nm)')
                elif num_common_waves == 1:
                     header['STEPCOMW'] = ('N/A', 'Step of common wavelengths (nm) - single point')

            # Try to get ACTPOS from source files
            try:
                # Look for any source file for this position to get ACTPOS
                source_pattern = os.path.join(self._storageFolder(), tt, f'image_*nm_pos{position:02d}.fits')
                source_files = glob.glob(source_pattern)
                
                if source_files:
                    # Read ACTPOS from the first available source file
                    with pyfits.open(source_files[0]) as source_hdu:
                        if 'ACTPOS' in source_hdu[0].header:
                            actpos = source_hdu[0].header['ACTPOS']
                            header['ACTPOS'] = (actpos, 'Actual position from source image')
                            self._logger.info(f"Added ACTPOS={actpos} from source file to fringe results header")
                        else:
                            self._logger.warning(f"ACTPOS not found in source file header: {source_files[0]}")
                else:
                    self._logger.warning(f"No source files found for position {position:02d} to get ACTPOS")
            except Exception as e:
                self._logger.error(f"Error reading ACTPOS from source files: {e}")

            hdul = pyfits.HDUList([primary_hdu])
            hdul.writeto(fringe_result_filename, overwrite=True)
            self._logger.info(f"Saved Qm data to {fringe_result_filename} with common wave info and ACTPOS.")
        except Exception as e:
             self._logger.error(f"Could not save Qm data to {fringe_result_filename}: {e}")
        # --- End Save Qm Data ---

        # --- Perform Matching based on selected method ---
        num_templates = Qt.shape[2]
        idp = -1
        idp_smooth = -1
        piston = np.nan
        piston_smooth = np.nan

        if matching_method == 'original':
            self._logger.info("Using 'original' matching method.")
            R = np.zeros(num_templates)
            R_smooth = np.zeros(num_templates)
            for i in range(num_templates):
                # Original calculation using dot product and norms
                qm_norm_sq = np.sum(Qm[:, :]**2)
                qt_norm_sq = np.sum(Qt[:, :, i]**2)
                if qm_norm_sq > 0 and qt_norm_sq > 0:
                    R[i] = np.sum(Qm[:, :] * Qt[:, :, i]) / (qm_norm_sq**0.5 * qt_norm_sq**0.5)
                else:
                    R[i] = -np.inf # Penalize if norm is zero

                qm_smooth_norm_sq = np.sum(Qm_smooth[:, :]**2)
                if qm_smooth_norm_sq > 0 and qt_norm_sq > 0:
                    R_smooth[i] = np.sum(Qm_smooth[:, :] * Qt[:, :, i]) / \
                                    (qm_smooth_norm_sq**0.5 * qt_norm_sq**0.5)
                else:
                    R_smooth[i] = -np.inf
            
            if np.any(np.isfinite(R)):
                 idp = np.nanargmax(R) # Use nanargmax to handle potential -inf
                 piston = np.atleast_1d(delta[idp]) # Ensure piston is at least 1D
            else:
                self._logger.warning("All 'original' R scores are non-finite. Cannot determine piston.")

            if np.any(np.isfinite(R_smooth)):
                idp_smooth = np.nanargmax(R_smooth)
                piston_smooth = np.atleast_1d(delta[idp_smooth]) # Ensure piston_smooth is at least 1D
            else:
                self._logger.warning("All 'original' R_smooth scores are non-finite. Cannot determine piston_smooth.")

        elif matching_method == 'cross_correlation':
            self._logger.info("Using 'cross_correlation' matching method.")
            R = np.zeros(num_templates)
            Qm_flat = Qm.flatten() # Flatten Qm once

            for i in range(num_templates):
                template_flat = Qt[:, :, i].flatten()
                
                # Check for zero variance which causes NaN in corrcoef
                if np.std(Qm_flat) < 1e-9 or np.std(template_flat) < 1e-9:
                    R[i] = -np.inf # Assign low score if variance is zero
                    continue
                
                try:
                    correlation_matrix = np.corrcoef(Qm_flat, template_flat)
                    # Check if result is valid (2x2 matrix and not NaN)
                    if correlation_matrix is not None and correlation_matrix.shape == (2, 2) and not np.isnan(correlation_matrix[0, 1]):
                         R[i] = correlation_matrix[0, 1]
                    else:
                        R[i] = -np.inf # Invalid result from corrcoef
                except ValueError as e:
                    # Catches cases like input contains NaN or infinity
                    self._logger.warning(f"np.corrcoef failed for template {i}: {e}")
                    R[i] = -np.inf

            if np.any(np.isfinite(R)):
                idp = np.nanargmax(R) # Use nanargmax to handle potential -inf
                piston = np.atleast_1d(delta[idp]) # Ensure piston is at least 1D
                piston_smooth = piston # Set smooth piston equal to piston (already 1D)
            else:
                self._logger.warning("All 'cross_correlation' scores are non-finite. Cannot determine piston.")
                piston_smooth = piston # Set to nan if piston is nan
        
        elif matching_method == 'template_matching':
            self._logger.info("Using 'template_matching' (skimage) method.")
            R = np.zeros(num_templates)

            # Pad Qm slightly so it's larger than the template for match_template
            # Pad width syntax: ((top, bottom), (left, right))
            pad_width = ((1, 1), (1, 1)) 
            try:
                Qm_padded = np.pad(Qm, pad_width=pad_width, mode='constant', constant_values=np.nan) 
                # Use nan padding and handle it later, or use edge/reflect? Constant 0 might skew results.
                # Replace NaNs from padding with a value that won't interfere (e.g., 0 if data is normalized 0-1) 
                # or handle potential NaNs in match_template if possible. Let's try replacing with 0 for now.
                Qm_padded = np.nan_to_num(Qm_padded, nan=0.0) 
            except Exception as e:
                self._logger.error(f"Failed to pad Qm for template matching: {e}")
                # Fallback or stop? Let's assign -inf to all scores for now.
                R.fill(-np.inf)
                idp = -1 # Ensure idp indicates failure
            else:
                # Proceed only if padding was successful
                for i in range(num_templates):
                    template_slice = Qt[:, :, i]
                    
                    # Ensure template is not all NaN or constant, which might cause issues
                    if np.isnan(template_slice).all() or np.ptp(template_slice) < 1e-9:
                        R[i] = -np.inf
                        continue
                        
                    try:
                        # match_template computes similarity; higher values are better matches.
                        match_result = match_template(Qm_padded, template_slice)
                        # The result is an array; the max value indicates the best match location/score.
                        if match_result.size > 0:
                            R[i] = np.nanmax(match_result) # Use nanmax just in case
                        else:
                            R[i] = -np.inf # Empty result
                    except Exception as e:
                        self._logger.warning(f"match_template failed for template {i}: {e}")
                        R[i] = -np.inf

            if np.any(np.isfinite(R)):
                idp = np.nanargmax(R) # Use nanargmax to handle potential -inf
                piston = np.atleast_1d(delta[idp]) # Ensure piston is at least 1D
                piston_smooth = piston # Set smooth piston equal to piston (already 1D)
            else:
                self._logger.warning("All 'template_matching' scores are non-finite. Cannot determine piston.")
                piston_smooth = piston # Set to nan if piston is nan

        # Add elif for other methods here in the future if needed
        
        else:
            self._logger.error(f"Unknown matching_method: '{matching_method}'. Cannot calculate piston.")
            # piston and piston_smooth remain NaN

        # --- End Matching ---
        
        # Print results only if piston was calculated
        # if not np.isnan(piston):
        #     print(f'Piston [nm] ({matching_method}) = ', piston * 1e9)
        # if not np.isnan(piston_smooth) and matching_method == 'original': # Only print smooth if different
        #      print('Piston smooth [nm] = ', piston_smooth * 1e9)
        # elif not np.isnan(piston_smooth) and matching_method != 'original':
        #      print(f'Piston_smooth [nm] ({matching_method}, set equal to piston) = ', piston_smooth * 1e9)

        # --- Plotting Best Match (if found) ---
        if idp != -1 and not np.isnan(piston).any(): # Check if a valid match was found
            try:
                best_template_slice = Qt[:, :, idp]
                target_name = f"Qm_pos{position:02d}"
                template_name = f"Template_Fringe_{idp+1:05d}" # Assuming template files are 1-indexed
                output_dir = os.path.join(self._storageFolder(), tt)
                plot_filename = os.path.join(output_dir, f"match_{target_name}_vs_{template_name}_{matching_method}.png")
                
                # Use common wavelengths for x-axis extent if available
                try: 
                    extent = [common_wavelengths.min(), common_wavelengths.max(), 0, Qm.shape[0]]
                except:
                    extent = None # Fallback if common_wavelengths is weird

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                
                # Qm is already normalized
                im1 = ax1.imshow(Qm, cmap='viridis', aspect='auto', origin='lower', extent=extent)
                ax1.set_title(f"Target: {target_name} ({matching_method})")
                ax1.set_xlabel("Wavelength (nm)" if extent else "Wavelength Index")
                ax1.set_ylabel("Pixel Row Index")
                fig.colorbar(im1, ax=ax1, label='Normalized Intensity')
                
                # Qt slices are already normalized
                im2 = ax2.imshow(best_template_slice, cmap='viridis', aspect='auto', origin='lower', extent=extent)
                title_str = f"Best Match: {template_name}"
                # Use piston[0] as it's now guaranteed to be at least 1D array
                title_str += f"\nPiston: {piston[0] * 1e9:.2f} nm"
                ax2.set_title(title_str)
                ax2.set_xlabel("Wavelength (nm)" if extent else "Wavelength Index")
                ax2.set_ylabel("Pixel Row Index")
                fig.colorbar(im2, ax=ax2, label='Normalized Intensity')
                
                plt.tight_layout()
                plt.savefig(plot_filename)
                plt.close(fig) 
                self._logger.info(f"Saved best match plot to {plot_filename}")

            except Exception as e:
                self._logger.error(f"Failed to generate or save best match plot for position {position:02d}: {e}")
        # --- End Plotting ---

        return piston, piston_smooth
    
    def _readDeltaAndLambdaFromFringesFolder(self, dove):
        dove_delta = os.path.join(dove, 'Differential_piston.fits')
        hduList = pyfits.open(dove_delta)
        delta = hduList[0].data
        dove_lambda_synth = os.path.join(dove, 'Lambda.fits')
        hduList = pyfits.open(dove_lambda_synth)
        
        lambda_synth_from_data = hduList[0].data
        # Convert synthetic wavelengths to nm if values are in meters (less than 1)
        if np.mean(lambda_synth_from_data) < 1:
            #print("Converting wavelengths from meters to nanometers")
            lambda_synth_from_data = lambda_synth_from_data * 1e9
        # Convert delta to meters if values are in nanometers (more than 1)
        if np.mean(np.abs(delta)) > 1:
            #print("Converting delta from nanometers to meters")
            delta = delta / 1e9
        return delta, lambda_synth_from_data

    def _myLambaSynth(self, lambda_synth_from_data):
        ''' Transforms its values into integers
        '''
        my_lambda = np.zeros(lambda_synth_from_data.shape[0])
        for j in range(lambda_synth_from_data.shape[0]):
            bb = int(round(lambda_synth_from_data[j]))
            my_lambda[j] = bb
        return my_lambda

    def _savePistonResult(self, tt, piston, piston_smooth, position):
        dove = os.path.join(self._storageFolder(), tt,
                            f'piston_result_pos{position:02d}.txt')
        file = open(dove, 'w+')
        file.write('%4e, %4e' %(piston[0], piston_smooth[0]))
        file.close()