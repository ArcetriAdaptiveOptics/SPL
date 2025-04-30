"""
Slices a full-frame FITS cube into subframes based on identified peaks.
Reads the raw data cube, identifies spot locations in a configured grid,
crops the 3D cube around each spot, and saves individual subframe cubes.
"""

import logging
import numpy as np
import os
from astropy.io import fits
import sys

# Using photutils for centroiding, similar to original analyzer
from photutils.centroids import centroid_com # Or centroid_2dg, centroid_1dg

# Import config using relative path
from ..conf import configuration as config

def _find_peak_centroid(image_2d, threshold=None):
    """Finds the centroid of the brightest region above a threshold.

    Args:
        image_2d (np.ndarray): The 2D image to analyze.
        threshold (float, optional): Minimum value threshold. If None, simple max is used.
                                     Consider adapting _findBrightSpots logic if needed.

    Returns:
        tuple or None: (col, row) coordinates of the centroid, or None if not found.
    """
    if threshold is not None:
        # Simple thresholding - may need refinement like original _findBrightSpots
        mask = image_2d < threshold
        if np.all(mask):
             #print("No pixels above threshold in subframe.")
             return None
        try:
            # centroid_com calculates the center of mass of the image above the mask
            col, row = centroid_com(image_2d, mask=mask)
            # print(f"Centroid found: col={col:.2f}, row={row:.2f}")
            return col, row
        except Exception as e:
             # Catch potential errors if centroiding fails (e.g., all masked)
             # print(f"Centroid calculation failed: {e}")
             return None
    else:
        # Fallback: find max pixel if no threshold provided (less robust)
        row, col = np.unravel_index(np.argmax(image_2d), image_2d.shape)
        # print(f"Max pixel found: col={col}, row={row}")
        return float(col), float(row)

def slice_cube(raw_cube_path, output_folder):
    """
    Loads a raw cube, finds peaks in subregions, crops around them, and saves subframe cubes.

    Args:
        raw_cube_path (str): Path to the raw FITS cube from acquisition (raw_cube.fits).
        output_folder (str): Path to the tracking number folder where outputs will be saved.

    Returns:
        list[str]: List of paths to the created subframe cube files.
    """
    logging.info(f"Slicing cube: {raw_cube_path}")

    # --- Read Configuration --- 
    try:
        slice_rows = getattr(config, 'SLICE_ROWS', 3) # Example default
        slice_cols = getattr(config, 'SLICE_COLS', 2) # Example default
        crop_height = getattr(config, 'CROP_HEIGHT', 200) # Example default (rows)
        crop_width = getattr(config, 'CROP_WIDTH', 300) # Example default (cols)
        # Default threshold is None, meaning use max pixel if not set in config
        spot_find_threshold = getattr(config, 'SPOT_FIND_THRESHOLD', None) 
        
        logging.info(f"Slicer Config: Grid={slice_rows}x{slice_cols}, Crop={crop_height}x{crop_width}, Threshold={spot_find_threshold}")
    except AttributeError as e:
        logging.error(f"Configuration error: Missing attribute in spl.conf.configuration: {e}")
        logging.error("Please ensure SLICE_ROWS, SLICE_COLS, CROP_HEIGHT, CROP_WIDTH are defined in configuration.py (SPOT_FIND_THRESHOLD is optional).")
        return []
    except Exception as e:
        logging.error(f"Error reading slicer configuration values: {e}")
        return []

    # --- Load Raw Data --- 
    try:
        with fits.open(raw_cube_path) as hdul:
            data_cube = hdul[0].data # Assumes (wavelength, height, width)
            header = hdul[0].header
            # Extract original wavelength list if needed later
            # wavelengths = eval(header.get('WAVELEN', '[]'))
        if data_cube.ndim != 3:
             logging.error(f"Expected 3D data cube, but got {data_cube.ndim} dimensions.")
             return []
        logging.info(f"Loaded raw cube with shape: {data_cube.shape}")
    except FileNotFoundError:
        logging.error(f"Raw cube file not found: {raw_cube_path}")
        return []
    except Exception as e:
        logging.error(f"Error loading raw cube file: {e}")
        return []

    # --- Prepare for Slicing --- 
    n_waves, full_height, full_width = data_cube.shape
    
    # Create 2D projection for spot finding (summing along wavelength axis)
    projection_2d = np.sum(data_cube, axis=0)
    logging.info(f"Created 2D projection for spot finding, shape: {projection_2d.shape}")

    subframe_height = full_height // slice_rows
    subframe_width = full_width // slice_cols
    
    subframe_paths = []
    position_index = 0

    # --- Loop Through Subframe Grid --- 
    logging.info("Scanning subframe grid for peaks...")
    for i in range(slice_rows):
        for j in range(slice_cols):
            # Define subframe boundaries on the 2D projection
            row_start = i * subframe_height
            row_end = (i + 1) * subframe_height
            col_start = j * subframe_width
            col_end = (j + 1) * subframe_width
            
            # Ensure boundaries don't exceed image dimensions on last row/col
            if i == slice_rows - 1:
                row_end = full_height
            if j == slice_cols - 1:
                col_end = full_width
            
            subframe_proj = projection_2d[row_start:row_end, col_start:col_end]
            
            # --- Find Peak Centroid in Subframe --- 
            peak_coords = _find_peak_centroid(subframe_proj, spot_find_threshold)
            
            if peak_coords is not None:
                local_col, local_row = peak_coords # Note: photutils returns (x,y) i.e. (col, row)
                
                # Convert local subframe centroid (col, row) to global image coordinates
                global_col = col_start + local_col
                global_row = row_start + local_row
                logging.info(f"  Peak found in subframe ({i},{j}) at global coords (col={global_col:.1f}, row={global_row:.1f})")
                
                # --- Crop Original 3D Cube --- 
                # Calculate crop boundaries centered on the global centroid
                # Ensure integers for slicing
                crop_row_start = max(0, int(np.round(global_row - crop_height / 2)))
                crop_row_end = min(full_height, int(np.round(global_row + crop_height / 2)))
                crop_col_start = max(0, int(np.round(global_col - crop_width / 2)))
                crop_col_end = min(full_width, int(np.round(global_col + crop_width / 2)))
                
                # Check if crop dimensions are valid
                if (crop_row_end <= crop_row_start) or (crop_col_end <= crop_col_start):
                    logging.warning(f"  Invalid crop dimensions for peak at ({global_col:.1f}, {global_row:.1f}). Skipping.")
                    continue
                    
                # Perform the 3D crop
                cropped_cube = data_cube[:, crop_row_start:crop_row_end, crop_col_start:crop_col_end]
                logging.info(f"    Cropped cube shape: {cropped_cube.shape}")

                # --- Save Cropped Cube --- 
                try:
                    output_filename = f"image_pos_{position_index:02d}.fits"
                    output_path = os.path.join(output_folder, output_filename)
                    
                    hdu = fits.PrimaryHDU(cropped_cube, header=header) # Copy original header
                    # Add/update relevant header info
                    hdu.header['SLICER'] = (__name__, 'Script that performed slicing')
                    hdu.header['POS_IDX'] = (position_index, 'Subframe position index')
                    hdu.header['CROP_R0'] = (crop_row_start, 'Crop start row (global)')
                    hdu.header['CROP_R1'] = (crop_row_end, 'Crop end row (global)')
                    hdu.header['CROP_C0'] = (crop_col_start, 'Crop start col (global)')
                    hdu.header['CROP_C1'] = (crop_col_end, 'Crop end col (global)')
                    hdu.header['PEAK_R'] = (global_row, 'Peak centroid row (global)')
                    hdu.header['PEAK_C'] = (global_col, 'Peak centroid col (global)')
                    # Keep original WAVELEN if it exists
                    
                    hdu.writeto(output_path, overwrite=True)
                    logging.info(f"    Saved cropped cube to: {output_path}")
                    subframe_paths.append(output_path)
                    position_index += 1 # Increment only when a cube is successfully saved
                except Exception as e:
                    logging.error(f"    Failed to save cropped cube for position {position_index}: {e}")
            # else: # No peak found in this subframe
            #    logging.debug(f"  No peak found in subframe ({i},{j}).")

    if not subframe_paths:
        logging.warning("Slicing complete, but no peaks were found or processed successfully.")
    else:
        logging.info(f"Slicing complete. Found and saved {len(subframe_paths)} subframe cubes.")
        
    return subframe_paths

if __name__ == '__main__':
    # Example usage for direct testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Fix for running script directly with relative imports ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..')) 
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added {project_root} to sys.path for direct script execution")
    # --- End of fix ---
    
    # --- Get paths for testing --- 
    # IMPORTANT: Modify these paths for your test case
    # Example: Use the latest folder in the measurement root
    try: 
        measurement_root = config.MEASUREMENT_ROOT_FOLDER
        latest_folder = max([os.path.join(measurement_root, d) for d in os.listdir(measurement_root) 
                            if os.path.isdir(os.path.join(measurement_root, d))], 
                            key=os.path.getmtime)
        raw_file = os.path.join(latest_folder, 'raw_cube.fits')
        out_dir = latest_folder
        print(f"Testing slicer with raw cube: {raw_file}")
        if os.path.exists(raw_file):
            sub_files = slice_cube(raw_file, out_dir)
            print(f"Slicing test completed. Created files: {sub_files}")
        else:
             print(f"ERROR: Test raw cube file not found: {raw_file}")

    except Exception as e:
        print(f"Error during slicer test setup or execution: {e}")
        logging.exception("Error detail:")

    # print("Run image_slicer.py directly for testing.") 