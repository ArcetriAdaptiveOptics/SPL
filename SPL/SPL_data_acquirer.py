'''
Authors
  - G. Pariani, R.Briguglio: written in 2016
  - C. Selmi: ported to Python in 2020
  - A. Puglisi and C. Selmi: python code debugging in 2021
'''

import os
import logging
import numpy as np
#from pathlib import Path
import scipy
import scipy.ndimage as scin
from astropy.io import fits as pyfits
from photutils import centroids
import matplotlib.pyplot as plt
from IPython.display import clear_output
from SPL.ground import tracking_number_folder

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
        self._pix2um = 3.75
        self._exptime = None

    @staticmethod
    def _storageFolder(path=None):
        """ Creates the path where to save measurement data"""
        #return Path(__file__).parent/'data'
        #return fold_name.SPL_ROOT_FOLDER
        lift_test_path = '/home/labot/LIFT/SPL/data'
        return lift_test_path


# lambda_vector = np.arange(530,730,10)
    def acquire(self, lambda_vector, exptime=0.7, mask=None):
        '''
        Parameters
        ----------
        lambda_vector: numpy array
                        vector of wavelengths (between 400/700 nm)

        Other Parameters
        ----------------
        exptime: float
            best exposure time for camera for laboratory conditions
        mask: numpy array
            mask for measurements
        Returns
        -------
        tt: string
            tracking number of measurements
        '''
        
        self._dove, tt = tracking_number_folder.createFolderToStoreMeasurements(self._storageFolder())
        fits_file_name = os.path.join(self._dove, 'lambda_vector.fits')
        pyfits.writeto(fits_file_name, lambda_vector)

        ## find PSF position ##
        print('Acquiring reference image at 600 nm...')
        self._filter.move_to(600)
        self._camera.setExposureTime(exptime/2 *1e3)
        aa = self._camera.exposureTime()
        print('with exposure time of %d [ms]' %aa)
        img = self._camera.getFutureFrames(1).toNumpyArray()
        if mask is None:
            mask = np.zeros(img.shape)

        reference_image = np.ma.masked_array(img, mask)

        if np.max(reference_image) > 4000:
            print("**************** WARNING: saturation detected!")
        # calcolo il baricentro
        cy, cx = self._baricenterCalculator(reference_image)


        expgain = np.ones(lambda_vector.shape[0]) * 0.5
        expgain[np.where(lambda_vector < 550)] = 1 #8
        expgain[np.where(lambda_vector > 650)] = 1 #3
        expgain[np.where(lambda_vector > 700)] = 1.5 #8

        self._logger.info('Acquisition of frames')

        for wl, expg in zip(lambda_vector, expgain):

            print('Acquiring image at %d nm...' % wl)
            self._filter.move_to(wl)
            self._camera.setExposureTime(exptime * expg *1e3)
            aa = self._camera.exposureTime()
            print('with exposure time of %d [ms]' %aa)
            #time.sleep(3 * ExposureTimeAbs / 1e6)
            image = self._camera.getFutureFrames(1).toNumpyArray()
            image = np.ma.masked_array(image, mask)
            #plot(image)
            #plt.imshow(image)
            #plt.show()

            crop = self._preProcessing(image, cy, cx)
            #crop_rot = scin.rotate(crop, 23,  reshape=False)
            #plt.figure(figsize=(10,5))
            #self.plot2(crop, crop_rot)

            file_name = 'image_%dnm.fits' %wl
            self._saveCameraFrame(file_name, crop)

        print('Saved tracking number:', tt)
        return tt

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
        return np.int(cy), np.int(cx)

#_baricenterCalculator
    def _preProcessing(self, image, cy, cx):
        ''' Cut the images around the pick position
        args:
            image = camera frame
        returns:
            crop = cut camera frame
        '''
        xcrop = 145 #150
        ycrop = 95 #100

        tmp = np.zeros((image.shape[0], image.shape[1]))
        tmp[cy-ycrop: cy+ycrop, cx-xcrop: cx+xcrop] = 1
        id_bkg = np.where(tmp == 0)
        bkg = np.ma.mean(image[id_bkg])
        img = image - bkg
        crop = img[cy-ycrop: cy+ycrop, cx-xcrop: cx+xcrop]
        return crop

    def _saveCameraFrame(self, file_name, frame):
        ''' Save camera frames in SPL/TrackingNumber
        '''
        fits_file_name = os.path.join(self._dove, file_name)
        pyfits.writeto(fits_file_name, frame.data)
        pyfits.append(fits_file_name, frame.mask.astype(np.int32))

    def plot(self, image):
        clear_output(wait=True)
        plt.imshow(image)
        plt.show()
        plt.pause(0.1)

    def plot2(self, img1, img2):
        clear_output(wait=True)
        plt.subplot(1,2,1)
        plt.imshow(img1)
        plt.subplot(1,2,2)
        plt.imshow(img2)
        plt.show()
        plt.pause(0.1)
    
#     def _saveInfo(self, file_name):
#         fits_file_name = os.path.join(self._dove, file_name)
#         header = pyfits.Header()
#         header['PIX2UM'] = self._pix2um
#
#     @staticmethod
#     def reloadSplAcquirer():
#         theObject = SPLAquirer(None, None)
#         theObject._pix2um = header['PIX2UM']
