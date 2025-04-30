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
from photutils.centroids import centroid_com
import cv2
import matplotlib.pyplot as plt


def calculate_psf_properties(data: np.ndarray, debug_save_contour: bool = False, output_prefix: str = 'debug') -> Optional[Tuple[float, Tuple[float, float]]]:
    """Calculates the orientation angle and centroid of the PSF using regionprops.

    Args:
        data (np.ndarray): The 2D PSF image.
        debug_save_contour (bool): If True, save a PNG showing the calculated contour.
        output_prefix (str): Prefix for debug output file names.

    Returns:
        Optional[Tuple[float, Tuple[float, float]]]: (angle_deg, centroid (row, col))
    """
    # Ensure input is 2D
    if data.ndim == 3:
        image_2d = np.sum(data, axis=0)
        logging.info(f"  Summed 3D data to 2D for property calculation.")
    elif data.ndim == 2:
        image_2d = data
    else:
        logging.error("  Cannot calculate properties for non 2D/3D data.")
        return None

    if np.all(image_2d == 0):
        logging.warning("  Image is all zeros, cannot calculate properties.")
        # Return 0 angle and geometric center as centroid if needed for fallback rotation
        center_row = (image_2d.shape[0] - 1) / 2.0
        center_col = (image_2d.shape[1] - 1) / 2.0
        return 0.0, (center_row, center_col)

    try:
        thresh = threshold_otsu(image_2d)
        binary = image_2d > thresh
        labeled_image = label(binary)
        regions = regionprops(labeled_image, intensity_image=image_2d)

        if not regions:
            logging.warning("  No regions found after thresholding. Cannot calculate properties.")
            return None

        largest_region = max(regions, key=lambda r: r.area)
        orientation_rad = largest_region.orientation
        centroid = largest_region.centroid # (row, col)
        logging.info(f"  Found largest region. Area: {largest_region.area}, Centroid: {centroid}")

        # Calculate angle (counter-clockwise from positive x-axis)
        angle_deg = 90.0 - np.degrees(orientation_rad)

        logging.info(f"  Calculated orientation (from y-axis, radians): {orientation_rad:.4f}")
        logging.info(f"  Converted angle (from x-axis, degrees): {angle_deg:.2f}")
        logging.info(f"  Calculated centroid (row, col): ({centroid[0]:.2f}, {centroid[1]:.2f})")

        # --- Debug: Save contour visualization --- 
        if debug_save_contour:
            try:
                # --- Prepare Background Image for Visualization ---
                # Use percentile clipping for more robust normalization against noise
                img_for_vis = image_2d.copy()
                p_low, p_high = np.percentile(img_for_vis, [1, 99]) # Use 1st and 99th percentiles
                img_clipped = np.clip(img_for_vis, p_low, p_high)
                
                # --- Convert to 8-bit grayscale for visualization ---
                min_val, max_val = np.min(img_clipped), np.max(img_clipped)
                if max_val > min_val:
                     vis_img_gray_8bit = (((img_clipped - min_val) / (max_val - min_val)) * 255).astype(np.uint8)
                else:
                     vis_img_gray_8bit = np.zeros_like(img_clipped, dtype=np.uint8) 
                     
                vis_img_rgb = cv2.cvtColor(vis_img_gray_8bit, cv2.COLOR_GRAY2BGR)
                # --- End Background Image Preparation ---

                # --- Find and Draw Contours ---
                # Convert the threshold mask to uint8 for findContours
                mask_uint8 = binary.astype(np.uint8) * 255 # Use the binary mask 
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw contours in green onto the visualization image (vis_img_rgb is BGR)
                cv2.drawContours(vis_img_rgb, contours, -1, (0, 255, 0), 1) # Draw green (BGR format)
                # --- End Find and Draw Contours ---
                
                # Save the image (already in BGR format)
                debug_filename = f"{output_prefix}_contour.png"
                cv2.imwrite(debug_filename, vis_img_rgb)
                logging.info(f"  Saved contour debug image to: {debug_filename}")
            except Exception as e:
                 logging.warning(f"  Could not save contour debug image: {e}")
        # --- End Debug ---

        return angle_deg, centroid

    except Exception as e:
        logging.error(f"  Error during property calculation: {e}")
        return None

def rotate_around_centroid(image: np.ndarray, angle: float, centroid: Tuple[float, float], order: int = 3, cval: float = 0.0) -> np.ndarray:
    """Rotates an image around a given centroid using shift-rotate-shift.
    
    Args:
        image (np.ndarray): The 2D image to rotate.
        angle (float): Rotation angle in degrees (counter-clockwise).
        centroid (Tuple[float, float]): Coordinates (row, col) of the rotation center.
        order (int): Interpolation order for shift and rotate.
        cval (float): Value used for points outside the boundaries.
        
    Returns:
        np.ndarray: The rotated image.
    """
    # Determine image center (rotation point for scipy.ndimage.rotate)
    image_center = np.array([(shape - 1) / 2.0 for shape in image.shape])
    
    # Calculate shift needed to move centroid to image center
    shift_to_center = image_center - np.array(centroid)
    
    # Pad image before shifting to avoid edge artifacts if centroid is near edge
    # (Calculate max necessary padding based on potential shift)
    max_shift = np.ceil(np.abs(shift_to_center)).astype(int)
    padded_image = np.pad(image, [(max_shift[0], max_shift[0]), (max_shift[1], max_shift[1])], mode='constant', constant_values=cval)
    effective_shift = shift_to_center + max_shift # Shift includes padding offset

    # 1. Shift the centroid (in the padded image) to the padded image's center
    shifted_img = shift(padded_image, shift=effective_shift, order=order, mode='constant', cval=cval)
    
    # 2. Rotate the shifted image around its center (which is now the centroid)
    rotated_img = rotate(shifted_img, angle, reshape=False, order=order, mode='constant', cval=cval)
    
    # 3. Shift the rotated image back by the negative of the effective shift
    final_img_padded = shift(rotated_img, shift=-effective_shift, order=order, mode='constant', cval=cval)
    
    # 4. Crop back to the original image size
    final_img = final_img_padded[max_shift[0]:-max_shift[0], max_shift[1]:-max_shift[1]]
    
    return final_img

def save_float_image_as_png(image: np.ndarray, filename: str):
    """Normalizes a float image using percentiles and saves as 8-bit PNG.
    
    Args:
        image (np.ndarray): Input float image.
        filename (str): Output PNG filename.
    """
    if image is None:
        logging.warning(f"Attempted to save None image to {filename}. Skipping.")
        return
    if not isinstance(image, np.ndarray):
         logging.warning(f"Invalid data type {type(image)} for PNG saving. Skipping {filename}.")
         return
         
    try:
        # Use percentile clipping for robust normalization
        p_low, p_high = np.percentile(image, [1, 99])
        img_clipped = np.clip(image, p_low, p_high)
        
        # Scale to 0-255 uint8
        min_val, max_val = np.min(img_clipped), np.max(img_clipped)
        if max_val > min_val:
             img_8bit = (((img_clipped - min_val) / (max_val - min_val)) * 255).astype(np.uint8)
        else:
             img_8bit = np.zeros_like(img_clipped, dtype=np.uint8)
             
        # Save using cv2
        cv2.imwrite(filename, img_8bit)
        logging.debug(f"Saved debug PNG: {filename}") # Use debug level for this potentially verbose output
    except Exception as e:
        logging.warning(f"Could not save debug PNG {filename}: {e}")

# Removed process_single_frame, pan_image_to_center, process_fits_file, main, and argparse

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
    parser.add_argument(
        "--no-pan",
        action="store_false", 
        dest="do_pan",       
        default=True,       
        help="Disable panning the image to the center."
    )
    parser.add_argument(
        "--debug-contour",
        action="store_true",
        default=False,
        help="Save a PNG showing the calculated contour used for centroiding."
    )

    args = parser.parse_args()

    input_path = args.input_file
    output_path = args.output_file
    user_angle = args.angle # User provided angle (or None)
    do_pan_flag = args.do_pan 
    debug_contour_flag = args.debug_contour # Get the debug flag

    # Check for scikit-image only if angle calculation is needed
    if user_angle is None:
        try:
            __import__('skimage')
            __import__('scipy') # Also needs scipy
        except ImportError as e:
            logging.error(f"Missing required library for automatic processing: {e}")
            logging.error("Please install scikit-image and scipy (`pip install scikit-image scipy`)")
            return
            
    process_single_frame(input_path, user_angle, debug_contour_flag, input_path)

if __name__ == "__main__":
    main() 