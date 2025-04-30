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
from spl.ground import smooth_function as sf
from spl.conf import configuration as config
import matplotlib.pyplot as plt
from astropy.io import fits
import re
from scipy.interpolate import interp1d


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
    
    def analyzer(self, tt, use_processed=False):
        '''
        Analyze measurement data and compare it with synthetic data.

        Parameters
        ----------
        tt: string
            tracking number of the measurement data
        use_processed: bool
            whether to use processed FITS files
        Returns
        -------
        pistons: list of tuples
            list of (piston, piston_smooth) for each position
        '''
        dove = os.path.join(self._storageFolder(), tt)

        self._logger.info('Analysis of tt = %s', tt)

        lambda_path = os.path.join(dove, 'lambda_vector.fits')
        hduList = pyfits.open(lambda_path)
        lambda_vector = hduList[0].data  # Assign lambda_vector here

        # Extract wavelengths and positions
        wavelengths, positions = self.parseFitsFilenames(dove)
        #print(wavelengths)
        #print(positions)

        pistons = []
        
        # For each position process the images
        for position in positions:
            cube, cube_normalized = self._readMeasurement(position, tt, use_processed)
            if cube is None: # Handle case where no files were found/read
                self._logger.warning(f"No valid data cube read for position {position}. Skipping analysis for this position.")
                pistons.append((np.nan, np.nan)) # Or some other indicator
                continue
            
            # Compute matrix and smoothed matrix
            matrix, matrix_smooth = self.matrix_calc(lambda_vector, cube, cube_normalized)

            # Save matrix
            self._saveMatrix(matrix, tt, wavelengths, position)

            # Compare with synthetic data to get piston and piston_smooth
            print('*** Position = ', position)
            piston, piston_smooth = self._templateComparison(matrix, matrix_smooth, lambda_vector, tt, position)

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

        for path in path_list:
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

        Args:
            matrix (numpy.ndarray): The matrix to save.
            tt (str): Tracking number.
            wavelength (int): Wavelength.
            position (int): Position.
        """
        destination_file_path = self._storageFolder()
        fits_file_name = os.path.join(destination_file_path, tt, f'fringe_result_pos{position:02d}.fits')
        pyfits.writeto(fits_file_name, matrix, overwrite=True)

    def matrix_calc(self, lambda_vector, cube, cube_normalized):
        img = np.sum(cube_normalized, 2)
        pick = self._newThr(img)
        matrix = np.zeros((pick[3]-pick[2] + 1, lambda_vector.shape[0])) # 150 + 1 pixel
        matrix_smooth = np.zeros((pick[3]-pick[2] + 1, lambda_vector.shape[0]))
        crop_frame_cube = None
        for i in range(lambda_vector.shape[0]):
            frame = cube[:, :, i]
            crop_frame = frame[pick[0]:pick[1], pick[2]:pick[3] + 1]
            #import code
            #code.interact(local=dict(globals(), **locals()))

            if crop_frame_cube is None:
                crop_frame_cube = crop_frame
            else:
                crop_frame_cube = np.dstack((crop_frame_cube, crop_frame))

            y = np.sum(crop_frame, 0)
            area = np.sum(y[:])
            y_norm = y / area
 #           if i == 0:
 #               mm = 1.2 * np.max(y_norm)
            matrix[:, i] = y_norm

            w = sf.smooth(y_norm, 4)
            w = w[:pick[3]-pick[2] + 1]
            matrix_smooth[:, i] = w

        matrix[np.where(matrix == np.nan)] = 0
        self._matrix = matrix
        self._matrixSmooth = matrix_smooth
        #self._saveMatrix(matrix)
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

    def _templateComparison(self, matrix, matrix_smooth, lambda_vector, tt, position):
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
        Returns
        -------
        piston: int
                piston value
        '''
        self._logger.debug('Template Comparison with data in tn_fringes = %s',
                           self.tn_fringes)
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
        for i in range(1, delta.shape[0]):
            file_name = os.path.join(dove, 'Fringe_%05d.fits' %i)
            hduList = pyfits.open(file_name)
            fringe = hduList[0].data
            fringe_selected = fringe[:, idx_synthetic]
            F.append(fringe_selected)
        F = np.dstack(F)
        Qt = F - np.mean(F)
        
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
            
            qm_plot_filename = os.path.join(self._storageFolder(), tt, f'Qm_plot_pos{position:02d}.png')
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
            
            qm_profiles_filename = os.path.join(self._storageFolder(), tt, f'Qm_profiles_pos{position:02d}.png')
            plt.savefig(qm_profiles_filename)
            plt.close() # Close plot to prevent display
            self._logger.info(f"Saved Qm profiles plot for position {position:02d} to {qm_profiles_filename}")
        except Exception as e:
             self._logger.error(f"Could not plot or save Qm profiles for position {position:02d}: {e}")
        # --- End Plot and Save Qm (1D Profiles) ---

        # --- Save Qm matrix data --- 
        try:
            qm_filename = os.path.join(self._storageFolder(), tt, f'Qm_matrix_pos{position:02d}.fits')
            pyfits.writeto(qm_filename, Qm, overwrite=True)
            self._logger.info(f"Saved Qm matrix data for position {position:02d} to {qm_filename}")
            # Optionally save common_wavelengths if needed with Qm data
            # pyfits.append(qm_filename, common_wavelengths) 
        except Exception as e:
             self._logger.error(f"Could not save Qm matrix data for position {position:02d}: {e}")
        # --- End Save Qm Data --- 

        R = np.zeros(delta.shape[0]-1)
        R_smooth = np.zeros(delta.shape[0]-1)
        for i in range(delta.shape[0]-1):
            R[i] = np.sum(Qm[:, :]*Qt[:, :, i]) / (np.sum(Qm[:, :]**2)**.5
                                                   * np.sum(Qt[:, :, i]**2)**.5)
            R_smooth[i] = np.sum(Qm_smooth[:, :]*Qt[:, :, i]) / \
                        (np.sum(Qm_smooth[:, :]**2)**.5 * np.sum(Qt[:, :, i]**2)**.5)

        idp = np.where(R == max(R))
        idp_smooth = np.where(R_smooth == max(R_smooth))
        
        piston = delta[idp]
        piston_smooth = delta[idp_smooth]
        print('Piston [nm] = ', piston * 1e9)
        print('Piston smooth [nm] = ', piston_smooth * 1e9)
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