'''
Authors
  - G. Pariani, R.Briguglio: written in 2016
  - C. Selmi: ported to Python in 2020
  - A. Puglisi and C. Selmi: python code debugging in 2021
  - M. Xompero and N. Azzaroli: modified SplAnalyzer from **5 to **.5
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



class SplAnalyzer_old():
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
    
    def captureDarkFrame(self, exptime):
        """Capture a dark frame with the same exposure time but with the lens closed."""
        print(f"Capturing dark frame with exposure time {exptime} ms...")

        # Move filter to the position where no light is being collected (closed filter)
        self._filter.move_to(-1)  # Assuming position 0 is closed, adjust if needed

        # Set the exposure time as in the rest of the program
        self._camera.setExposureTime(exptime)

        # Capture a single dark frame (without averaging multiple frames)
        dark_frame = self._camera.getFutureFrames(0).toNumpyArray()

        # In case multiple frames are returned, average them (optional)
        dark_frame_mean = np.mean(dark_frame, axis=-1)  # Averaging, but you can skip if only one frame is captured
    
        print(f"Dark frame captured.")
        return dark_frame_mean


    def analyzer(self, tt):
        '''
        Analyze measurement data and compare it with synthetic data.

        Parameters
        ----------
        tt: string
            tracking number of the measurement data
        Returns
        -------
        piston: int
                piston value
        piston_smooth: int
                piston value after smoothing data
        '''
        dove = os.path.join(self._storageFolder(), tt)
        #print('Analyzing ',dove)
        self._logger.info('Analysis of tt = %s', tt)

        lambda_path = os.path.join(dove, 'lambda_vector.fits')
        hduList = pyfits.open(lambda_path)
        lambda_vector = hduList[0].data
        #print('Lambda vector = ', lambda_vector)

        cube, cube_normalized = self.readMeasurement(tt)
        print(f"Cube shape: {cube.shape}")
        matrix, matrix_smooth = self.matrix_calc(lambda_vector, cube, cube_normalized)

        # plt.imshow(matrix, cmap='viridis', aspect='auto')  # Usa una mappa di colori adatta
        # plt.colorbar(label='Intensity')  # Aggiunge una barra dei colori
        # plt.xlabel('Wavelength index')  # Etichetta asse X
        # plt.ylabel('Pixel position index')  # Etichetta asse Y
        # plt.title('Matrix Visualization')
        # plt.show()
            
        self._saveMatrix(matrix, tt)
        piston, piston_smooth = self._templateComparison(matrix,
                                                         matrix_smooth,
                                                         lambda_vector)

        self._savePistonResult(tt, piston, piston_smooth)

        return piston, piston_smooth
    ### PLOT MATRIX ###
    # x = lambda_vector
    # y = np.arange(151)* spl._pix2um
    # imshow(spl._matrix, extent = [x[0],x[19],y[0], y[150]], origin= 'lower');
    #colorbar(); plt.xlabel('lambda [nm]'); plt.ylabel('position [um]');
    #plt.title('TN = %s' %tt)

    def _saveMatrix(self, matrix, tt):
        destination_file_path = self._storageFolder()
        fits_file_name = os.path.join(destination_file_path,tt,
                                      'fringe_result.fits')
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

    def readMeasurement(self, tt):
        """ Read images from a specific tracking number and return the cube.

        Args:
            tt (str): Tracking number.

        Returns:
            tuple: 
                - cube (numpy.ndarray): [pixels, pixels, n_frames]
                - cube_normalized (numpy.ndarray): Normalized images, [pixels, pixels, n_frames]
        """
        dove = os.path.join(self._storageFolder(), tt)
        
        # Get sorted list of FITS files
        path_list = sorted(glob.iglob(os.path.join(dove, 'image_*.fits')))
        
        if not path_list:
            print(f"Error: No FITS files found in {dove}")
            return None, None  # Handle case with no images

        cube = []
        cube_normalized = []

        for path in path_list:
            #print(f"Reading {path}")
            
            try:
                with pyfits.open(path) as hduList:
                    image = hduList[0].data
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue  # Skip corrupted files

            if image is None:
                print(f"Warning: No data in {path}")
                continue  # Skip empty images

            image_sum = np.sum(image)
            image_normalized = image / image_sum if image_sum != 0 else image  # Prevent division by zero
            
            cube.append(image)
            cube_normalized.append(image_normalized)

        if not cube:
            print("Error: No valid images found!")
            return None, None  # Handle case where all files were skipped

        # Stack images along the third dimension
        return np.dstack(cube), np.dstack(cube_normalized)
    
    
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
                baricenterCoord[1]-75, baricenterCoord[1]+75]
        return pick

    def _templateComparison(self, matrix, matrix_smooth, lambda_vector):
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
        Returns
        -------
        piston: int
                piston value
        '''
        self._logger.debug('Template Comparison with data in tn_fringes = %s',
                           self.tn_fringes)
        dove = os.path.join(self._storageFringesFolder(),
                            self.tn_fringes)
        #print('Restoring synth data from: ',dove)
        delta, lambda_synth_from_data = self._readDeltaAndLambdaFromFringesFolder(dove)
        lambda_synth = self._myLambaSynth(lambda_synth_from_data)
        #print('Lambda synth = ', lambda_synth)

        idx = np.isin(lambda_synth, lambda_vector)
        #print('Synth lambdas = ', lambda_synth)

        Qm = matrix - np.mean(matrix)
        # plt.imshow(Qm.T); plt.title('SPL Signal')
        # plt.show()

        self._Qm = Qm
        Qm_smooth = matrix_smooth - np.mean(matrix_smooth)
        self._QmSmooth = Qm_smooth
        #creare la matrice di Giorgio della giusta dimensione
        F = []
        for i in range(1, delta.shape[0]):
            file_name = os.path.join(dove, 'Fringe_%05d.fits' %i)
            #print('Reading ',file_name)
            hduList = pyfits.open(file_name)
            fringe = hduList[0].data
            fringe_selected = fringe[:, idx]
            F.append(fringe_selected)
        F = np.dstack(F)
        Qt = F - np.mean(F)
        self._Qt = Qt

        R = np.zeros(delta.shape[0]-1)
        R_smooth = np.zeros(delta.shape[0]-1)
        for i in range(delta.shape[0]-1):
            # print('Processing',i)
            R[i] = np.sum(Qm[:, :]*Qt[:, :, i]) / (np.sum(Qm[:, :]**2)**.5
                                                   * np.sum(Qt[:, :, i]**2)**.5)
            R_smooth[i] = np.sum(Qm_smooth[:, :]*Qt[:, :, i]) / \
                        (np.sum(Qm_smooth[:, :]**2)**.5 * np.sum(Qt[:, :, i]**2)**.5)

        # plt.plot(R); plt.title('R')
        # plt.show()

        idp = np.where(R == max(R))
        idp_smooth = np.where(R_smooth == max(R_smooth))
        plt.imshow(F[:, :, idp[0][0]].T); plt.title('Synth fringes')
        plt.show()
        
        piston = delta[idp]
        piston_smooth = delta[idp_smooth]
        #print('Piston = ', piston)
        #print('Piston smooth = ', piston)
        return piston, piston_smooth
    
    def _readDeltaAndLambdaFromFringesFolder(self, dove):
        dove_delta = os.path.join(dove, 'Differential_piston.fits')
        hduList = pyfits.open(dove_delta)
        delta = hduList[0].data
        dove_lambda_synth = os.path.join(dove, 'Lambda.fits')
        hduList = pyfits.open(dove_lambda_synth)
        lambda_synth_from_data = hduList[0].data * 1e9
        return delta, lambda_synth_from_data

    def _myLambaSynth(self, lambda_synth_from_data):
        ''' Transforms its values into integers
        '''
        my_lambda = np.zeros(lambda_synth_from_data.shape[0])
        for j in range(lambda_synth_from_data.shape[0]):
            bb = int(round(lambda_synth_from_data[j]))
            my_lambda[j] = bb
        return my_lambda

    def _savePistonResult(self, tt, piston, piston_smooth):
        dove = os.path.join(self._storageFolder(), tt,
                            'piston_result_prova.txt')
        file = open(dove, 'w+')
        file.write('%4e, %4e' %(piston[0], piston_smooth[0]))
        file.close()
