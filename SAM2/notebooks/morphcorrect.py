import os
import numpy as np
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes, zoom, gaussian_filter
import tifffile as tiff
import multiprocessing as mp

def size_filter(slice, min_size):
    """Remove small objects in the slice."""
    labeled_slice, _ = label(slice, connectivity=1, return_num=True)
    props = regionprops(labeled_slice)

    for prop in props:
        
        if prop.area < min_size:
            print(prop.area)
            labeled_slice[labeled_slice == prop.label] = 0

    return (labeled_slice > 0).astype(np.uint8) * 255

def process_single_slice(file_path):
    """Process a single mask: fill holes and apply size filtering."""
    mask = imread(file_path)
    mask_binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    filled_mask = binary_fill_holes(mask_binary // 255) * 255
    filtered_mask = size_filter(filled_mask, min_size=100)
    return filtered_mask


def smooth_z_axis(volume, sigma=1.0):
    """Apply Gaussian smoothing along the z-axis to reduce sharp transitions."""
    smoothed_volume = gaussian_filter(volume, sigma=[sigma, 0, 0])
    return smoothed_volume

def process_slices(input_folder, output_folder):
    """Process all slices in the input folder and stack into a volume."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Collect and sort all .png files in the input folder
    files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
    file_paths = [os.path.join(input_folder, f) for f in files]
    
    
    with mp.Pool(6) as pool:
        cleaned_masks = pool.map(process_single_slice, file_paths)

    for filename, cleaned_mask in zip(files, cleaned_masks):
        output_path = os.path.join(output_folder, filename)
        imsave(output_path, cleaned_mask)
        print(f"Processed and saved: {output_path}")

    volume = np.stack(cleaned_masks, axis=0)

    return volume

def interpolate_slices(volume, zoom_factor):
    """
    Interpolate along the z-axis using scipy.ndimage.zoom.
    
    Args:
    - volume: 3D numpy array (z, x, y) representing the volume.
    - zoom_factor: Float, how much to zoom the z-axis.
    
    Returns:
    - Interpolated volume with adjusted z-dimension.
    """
    # Apply zoom along the z-axis, while keeping x and y dimensions unchanged
    interpolated_volume = zoom(volume, (zoom_factor, 1, 1), order=3)
    
    return interpolated_volume

def main():
    input_folder = '/home/codee/scratch/segment-anything-2/predictedpv'
    output_folder = '/home/codee/scratch/segment-anything-2/predictedpv_refined'
    output_tiff_path = '/home/codee/scratch/segment-anything-2/volume_pv.tiff'
    
    # Process slices and stack them into a volume
    cleaned_volume = process_slices(input_folder, output_folder)

    downsamplescale = 0.05
    original_z_resolution = 80  # nm
    desired_z_resolution = 8  # nm
    zoom_factor = original_z_resolution / desired_z_resolution * downsamplescale
    
    # Interpolate the volume
    interpolated_volume = zoom(cleaned_volume, (zoom_factor, 1, 1), order=3)

    # Apply Gaussian smoothing along the z-axis
    smoothed_volume = smooth_z_axis(interpolated_volume, sigma=1.0)
    # Convert to binary (threshold at 127.5)
    binary_volume = np.where(smoothed_volume > 127.5, 255, 0).astype(np.uint8)
    
    tiff.imwrite(output_tiff_path, binary_volume)
    print(f'Interpolated volume saved as {output_tiff_path}')

if __name__ == "__main__":
    main()
