'''
Script to rotate and pan a 2D PSF array or 3D cube stored in a FITS file.
Rotation angle can be provided or estimated via ellipse fitting.
Panning shifts the calculated PSF centroid to the image center.
'''

import argparse
import logging
import os
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits
from scipy.ndimage import rotate, shift
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
# Import photutils centroiding
from photutils.centroids import centroid_com


def calculate_psf_properties(data: np.ndarray) -> Optional[Tuple[float, Tuple[float, float]]]:
    """Calculates the orientation angle and centroid of the PSF by fitting an ellipse.

    If the data is 3D, it first sums along the first axis.

    Args:
        data (np.ndarray): The 2D PSF image or 3D cube.

    Returns:
        Optional[Tuple[float, Tuple[float, float]]]: 
            A tuple containing:
            - angle (float): Calculated angle in degrees (counter-clockwise from x-axis).
            - centroid (tuple): Calculated centroid (row, col).
            Returns None if fitting fails.
    """
    logging.info("Attempting to calculate PSF angle and centroid...")
    if data.ndim == 3:
        image_2d = np.sum(data, axis=0)
        logging.info(f"  Summed 3D data (shape {data.shape}) to 2D (shape {image_2d.shape}) for fitting.")
    elif data.ndim == 2:
        image_2d = data
    else:
        logging.error("  Cannot calculate properties for data that is not 2D or 3D.")
        return None

    if np.all(image_2d == 0):
        logging.warning("  Image is all zeros, cannot calculate properties.")
        # Return 0 angle, but None centroid (treat as centered, no panning needed/possible)
        center_row = (image_2d.shape[0] - 1) / 2.0
        center_col = (image_2d.shape[1] - 1) / 2.0
        return 0.0, (center_row, center_col) # Return 0 angle and geometric center

    try:
        # --- Centroid Calculation (using centroid_com) ---
        thresh = threshold_otsu(image_2d)
        # Create a mask where True indicates pixels *above* the threshold
        threshold_mask = image_2d > thresh 
        if not np.any(threshold_mask):
            logging.warning("  No pixels found above Otsu threshold. Cannot calculate centroid_com.")
            centroid = None
        else:
            try:
                # Calculate centroid_com using only pixels above threshold
                # centroid_com mask: True means masked/excluded
                com_centroid = centroid_com(image_2d, mask=~threshold_mask)
                centroid = (com_centroid[1], com_centroid[0]) # photutils returns (x,y), we use (row,col)
                logging.info(f"  Calculated centroid_com (row, col): ({centroid[0]:.2f}, {centroid[1]:.2f})")
            except Exception as e:
                logging.warning(f"  Error during centroid_com calculation: {e}. Proceeding without centroid.")
                centroid = None
        
        # --- Orientation/Angle Calculation (using regionprops) ---
        angle_deg = 0.0 # Default angle
        try:
            labeled_image = label(threshold_mask) # Use the same threshold mask
            regions = regionprops(labeled_image) # No intensity image needed for orientation

            if regions:
                largest_region = max(regions, key=lambda r: r.area)
                # Use centroid from regionprops only if com fails? For now, prioritize com.
                # region_centroid = largest_region.centroid # (row, col)
                orientation_rad = largest_region.orientation
                angle_deg = 90.0 - np.degrees(orientation_rad)
                logging.info(f"  Found largest region. Area: {largest_region.area}")
                logging.info(f"  Calculated orientation (from y-axis, radians): {orientation_rad:.4f}")
                logging.info(f"  Converted angle (from x-axis, degrees): {angle_deg:.2f}")
            else:
                logging.warning("  No regions found after thresholding for orientation calculation. Using angle 0.")
        except Exception as e:
             logging.warning(f"  Error during regionprops/orientation calculation: {e}. Using angle 0.")
             
        # Return results (prioritize centroid_com if available)
        if centroid is not None:
            return angle_deg, centroid
        else:
            # If centroid failed, don't return anything, indicating failure
            logging.warning("  Failed to calculate a valid centroid.")
            return None

    except Exception as e:
        logging.error(f"  Error during property calculation: {e}")
        return None

def pan_image_to_center(image: np.ndarray, centroid: Tuple[float, float]) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Shifts a 2D image so the centroid is at the geometric center.

    Args:
        image (np.ndarray): The 2D image to pan.
        centroid (Tuple[float, float]): The coordinates (row, col) of the centroid.

    Returns:
        Tuple[np.ndarray, Tuple[float, float]]: 
            - The panned 2D image.
            - The applied shift (shift_row, shift_col).
    """
    center_row = (image.shape[0] - 1) / 2.0
    center_col = (image.shape[1] - 1) / 2.0
    
    shift_row = center_row - centroid[0]
    shift_col = center_col - centroid[1]
    applied_shift = (shift_row, shift_col)

    logging.info(f"    Panning image. Center: ({center_row:.2f}, {center_col:.2f}), Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}), Shift: ({shift_row:.2f}, {shift_col:.2f})")
    
    # Use scipy.ndimage.shift for subpixel precision
    panned_image = shift(image, shift=applied_shift, order=3, mode='constant', cval=0.0)
    
    return panned_image, applied_shift


def process_fits_file(input_path: str, output_path: str, user_angle: Optional[float], do_pan: bool = True):
    """Loads FITS data, calculates properties, rotates, optionally pans to center, and saves.

    Args:
        input_path (str): Path to the input FITS file (2D/3D).
        output_path (str): Path to save the processed FITS file.
        user_angle (Optional[float]): Rotation angle in degrees (counter-clockwise).
                                     If None, calculates orientation automatically.
        do_pan (bool): If True, pan the image to center the calculated centroid. Defaults to True.
    """
    try:
        logging.info(f"Loading data from: {input_path}")
        with fits.open(input_path) as hdul:
            if hdul[0].data is None:
                logging.error("No data found in the primary HDU.")
                return
            data = hdul[0].data.astype(np.float32)
            header = hdul[0].header

        # Calculate PSF properties (angle and centroid)
        properties = calculate_psf_properties(data)
        calculated_angle = None
        centroid = None
        if properties:
            calculated_angle, centroid = properties
        else:
            logging.warning("Could not calculate PSF properties (angle/centroid).")

        # Determine final angle for rotation
        final_angle = user_angle
        auto_calculated_angle = False
        if final_angle is None:
            if calculated_angle is not None:
                final_angle = calculated_angle
                auto_calculated_angle = True
                logging.info(f"Using calculated angle for rotation: {final_angle:.2f} degrees")
            else:
                # If user didn't provide angle and calculation failed, use 0 rotation but still try to pan if centroid exists
                logging.warning("Rotation angle not provided and automatic calculation failed. Using 0 degrees rotation.")
                final_angle = 0.0 
        else:
            logging.info(f"Using provided angle for rotation: {final_angle:.2f} degrees")

        # Log the final angle before applying rotation
        logging.info(f"Final rotation angle to be applied: {final_angle:.2f} degrees")
        
        # --- Rotation Step ---            
        rotated_data = data # Start with original if rotation fails or is 0
        if abs(final_angle) > 1e-6: # Avoid rotating by 0
            logging.info(f"Rotating data by {final_angle:.2f} degrees...")
            if data.ndim == 2:
                rotated_data = rotate(data, final_angle, reshape=False, order=3, mode='constant', cval=0.0)
            elif data.ndim == 3:
                rotated_slices = []
                num_slices = data.shape[0]
                for i in range(num_slices):
                    slice_2d = data[i, :, :]
                    rotated_slice = rotate(slice_2d, final_angle, reshape=False, order=3, mode='constant', cval=0.0)
                    rotated_slices.append(rotated_slice)
                    if (i + 1) % 10 == 0 or (i + 1) == num_slices:
                        logging.info(f"  Rotated slice {i+1}/{num_slices}")
                rotated_data = np.stack(rotated_slices, axis=0)
            else:
                logging.error(f"Input data has unsupported dimensions: {data.ndim}. Cannot rotate.")
                return # Or handle differently?
            logging.info(f"Rotation complete. Rotated data shape: {rotated_data.shape}")
        else:
             logging.info("Skipping rotation (angle is effectively 0).")

        # --- Panning Step --- 
        final_data = rotated_data # Start with rotated (or original if no rotation) data
        applied_shift = None
        if do_pan:
            if centroid:
                logging.info("Centroid calculated. Proceeding with panning to center...")
                if rotated_data.ndim == 2:
                    final_data, applied_shift = pan_image_to_center(rotated_data, centroid)
                elif rotated_data.ndim == 3:
                    # Pan each slice using the same shift calculated from the overall centroid
                    panned_slices = []
                    num_slices = rotated_data.shape[0]
                    # Calculate shift once based on the first slice dimensions (assuming all are same)
                    # Use the centroid calculated from the original data
                    temp_panned, current_shift = pan_image_to_center(rotated_data[0,:,:], centroid)
                    applied_shift = current_shift # Store the shift calculated
                    panned_slices.append(temp_panned)
                    logging.info(f"  Calculated panning shift (row, col): ({applied_shift[0]:.2f}, {applied_shift[1]:.2f}) for all slices")
                    logging.info(f"    Panned slice 1/{num_slices}")
                    # Apply the same shift to remaining slices
                    for i in range(1, num_slices):
                        rotated_slice = rotated_data[i, :, :]
                        panned_slice = shift(rotated_slice, shift=applied_shift, order=3, mode='constant', cval=0.0)
                        panned_slices.append(panned_slice)
                        if (i + 1) % 10 == 0 or (i + 1) == num_slices:
                             logging.info(f"    Panned slice {i+1}/{num_slices}")
                    final_data = np.stack(panned_slices, axis=0)
                logging.info(f"Panning complete. Final data shape: {final_data.shape}")
            else:
                # Only log warning if panning was requested but couldn't happen
                logging.warning("No centroid calculated or found. Skipping panning step even though requested.")
        else:
             # Log info if user explicitly skipped panning
             logging.info("Skipping panning step as requested by user.")

        # --- Update Header --- 
        header['HISTORY'] = f"Processed by process_psf script."
        header['HISTORY'] = f"Applied rotation of {final_angle:.2f} degrees."
        if auto_calculated_angle:
             header['HISTORY'] = "Rotation angle was automatically calculated."
        else:
             header['HISTORY'] = "Rotation angle was user-provided."
        # Update panning history messages
        if applied_shift:
             header['HISTORY'] = f"Applied panning shift (row, col): ({applied_shift[0]:.2f}, {applied_shift[1]:.2f})."
        elif do_pan and not centroid:
             header['HISTORY'] = "Panning requested but skipped (no centroid)."
        elif not do_pan:
             header['HISTORY'] = "Panning was explicitly skipped."
        if centroid:
            header['CENTROID'] = (f"{centroid[0]:.2f},{centroid[1]:.2f}", 'Calculated centroid (row, col) before pan')
        header['ROTANGLE'] = (final_angle, 'Applied rotation angle in degrees')
        header['ROTAUTO'] = (auto_calculated_angle, 'Was rotation angle calculated automatically?')
        if applied_shift:
             header['PANSHIFT'] = (f"{applied_shift[0]:.2f},{applied_shift[1]:.2f}", 'Applied shift (row, col) to center')

        # --- Save Result --- 
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")

        logging.info(f"Saving final processed data to: {output_path}")
        hdu = fits.PrimaryHDU(final_data, header=header)
        hdul_out = fits.HDUList([hdu])
        hdul_out.writeto(output_path, overwrite=True)
        logging.info("File saved successfully.")

    except FileNotFoundError:
        logging.error(f"Input file not found: {input_path}")
    except ImportError as e:
        # Check if the missing import is skimage specifically
        if 'skimage' in str(e) and user_angle is None:
            logging.error("Scikit-image is required for automatic angle detection. Please install it (`pip install scikit-image`)")
        else:
            logging.error(f"An import error occurred: {e}") # Log other import errors
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        raise

def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(
        description="Rotate & Pan a 2D/3D FITS file. Rotation angle is optional (auto-calculated). Panning centers the PSF."
    )
    parser.add_argument("input_file", help="Path to the input FITS file.")
    parser.add_argument("output_file", help="Path to save the processed FITS file.")
    parser.add_argument(
        "angle", 
        type=float, 
        nargs='?', 
        default=None,
        help="Rotation angle in degrees (counter-clockwise). If omitted, calculates automatically."
    )
    # Add --no-pan argument
    parser.add_argument(
        "--no-pan",
        action="store_false", # Sets dest to False if flag is present
        dest="do_pan",       # The variable name to store the result
        default=True,       # Default value if flag is NOT present
        help="Disable panning the image to the center."
    )

    args = parser.parse_args()

    input_path = args.input_file
    output_path = args.output_file
    user_angle = args.angle # User provided angle (or None)
    do_pan_flag = args.do_pan # Get the boolean flag from args

    # Check for scikit-image only if angle calculation is needed
    if user_angle is None:
        try:
            __import__('skimage')
            __import__('scipy') # Also needs scipy
        except ImportError as e:
            logging.error(f"Missing required library for automatic processing: {e}")
            logging.error("Please install scikit-image and scipy (`pip install scikit-image scipy`)")
            return
            
    process_fits_file(input_path, output_path, user_angle, do_pan_flag)

if __name__ == "__main__":
    main() 