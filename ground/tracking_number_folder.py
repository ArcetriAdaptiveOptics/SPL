'''
Authors
  - C. Selmi: written in 2019
             modified in July 2021

HOW TO USE IT::

    from SPL.ground import tracking_numbers_folder as tnf
    path, tt = tnf.createFolderToStoreMeasurements(store_in_folder)
    or
    path = tnf.findTrackingNumberPath(tt)
'''

import os
import fnmatch
from SPL.ground.timestamp import Timestamp

def createFolderToStoreMeasurements(store_in_folder):
    """
    Create a new folder using the generation of the tracking number

    Parameters
    ----------
    store_in_folder: string
                    path where to create new fold

    Returns
    -------
        dove: string
            all path generated
        tt: string
            only tracking number
    """
    tt = Timestamp.now()
    dove = os.path.join(store_in_folder, tt)
    if os.path.exists(dove):
        raise OSError('Directory %s exists' % dove)
    else:
        os.makedirs(dove)
    return dove, tt

def findTrackingNumberPath(tt):
    ''' Function to find the entire path from tracking number

    Parameters
    ----------
        tt: string
            tracking number

    Returns
    -------
        final_path: string
            total path
    '''
    rootPath = '?'
    if rootPath is None:
        raise OSError('No configuration has been loaded!')
    pattern = tt

    for root, dirs, files in os.walk(rootPath):
        for directory in fnmatch.filter(dirs, pattern):
            final_path = os.path.join(root, directory)
    try:
        return final_path
    except:
        raise OSError('Tracking number %s does not exists' % tt)
