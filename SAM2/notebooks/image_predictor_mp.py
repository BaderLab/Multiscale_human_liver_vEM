"""
This file is to use the SAM2 predictor class to predict the instance.
The prompt is the dot. Though for each single image there can be 
more than one dot as the prompt, each prediction is only based on 
one single dot and the predicted masks are integrated together as one mask
if there are multiple dots. 

The script is programmed with multiprocessing to speed up.

The prediction is based on the downsampled image with the size (1000*1000). 
"""

import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import hydra
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra
from skimage import transform
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from multiprocessing import Pool, cpu_count, set_start_method
Image.MAX_IMAGE_PIXELS = 500000000

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    if borders:
        '''
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
        '''
    #upsample_factor = 10
    #mask_image = transform.rescale(mask_image, upsample_factor, anti_aliasing=True, channel_axis=-1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = np.array(coords[labels==1])
    neg_points = np.array(coords[labels==0])
    pos_points = pos_points.reshape(-1, 2)
    neg_points = neg_points.reshape(-1, 2)
    #print(pos_points)
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        if image.dtype not in [np.float32, np.float64, np.uint8]:
            image = image.astype(np.uint8)
        plt.imshow(image, cmap='gray')
        show_mask(mask, plt.gca(), borders=borders)
        #print(point_coords, input_labels)

        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        #plt.axis('off')
        #plt.savefig(f"/home/codee/scratch/segment-anything-2/notebooks/images/predicted_mp_{i}.png")
        #plt.show()

def find(slice_id, input_em_dir):
    """
    Utility function to find the image corresponding to the slice_id
    within the input_em_dir. This assumes filenames contain the slice number.
    """
    for i, filename in enumerate(sorted(os.listdir(input_em_dir))):
        if f"{slice_id:03d}" in filename:
            return i
    return None

def slicenumber(image_file):
    """
    Extracts the slice number from the filename. Adjust this according to your file naming format.
    """
    # in the form of human_100_0000, so id is the second last '100'
    return int(os.path.basename(image_file).split('_')[-2].split('.')[0])

from PIL import Image

def save_mask_as_image(mask, save_path):
    mask_img = (mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_img)
    mask_pil.save(save_path)

def combine_masks(mask1, mask2):
    """
    Combines two masks using logical OR operation (i.e., union of both masks).
    Ensure that the masks are of the same shape.
    """
    return np.logical_or(mask1, mask2).astype(np.uint8)

def get_batch(input_em_dir, input_swc_file):
    """
    Aims to get the batch points for each image within input_em_dir.

    Returns: points_batch (list of points for each image),
             images_batch (list of image arrays),
             zslice_id_batch (list of slice numbers).
    """
    points_batch = [[] for _ in range(len(os.listdir(input_em_dir)))]  # Initialize list for storing points
    with open(input_swc_file, 'r') as swc_file:
        for line in swc_file:
            if line.startswith('#'):
                continue  # Skip header lines
            line = list(map(float, line.strip().split()))  # Parse SWC file
            # Extract slice id and convert to file scale
            slice_id = int(line[4] * 20)  # Adjust according to scale
            id_in_dir = find(slice_id, input_em_dir)
            if id_in_dir is not None:
                points_batch[id_in_dir].append([line[2] * 1000 / 8 / 20, line[3] * 1000 / 8 / 20])

    images_batch = []
    zslice_id_batch = []
    for id, image_file in enumerate(sorted(os.listdir(input_em_dir))):
        if len(points_batch[id]) == 0:
            continue  # Skip if no points for this image
        image_path = os.path.join(input_em_dir, image_file)
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        images_batch.append(image)
        #print(slicenumber(image_file))
        zslice_id_batch.append(slicenumber(image_file))

    return points_batch, images_batch, zslice_id_batch

def process_single_image(args):
    """
    This function processes a single image, predicts masks for each point separately,
    and combines them into a single mask.
    """
    image, points, zslice_id = args

    # Reinitialize the SAM2 model and predictor inside the process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam2_checkpoint = "/home/codee/scratch/segment-anything-2/checkpoints/sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    # Initialize an empty mask with the same shape as the image
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Predict masks for each point one by one
    for i, point in enumerate(points):
        point_label = np.array([1])  # Assuming each point is a positive point
        single_point = np.array([point])

        try:
            predictor.set_image(image)
            mask, scores, _ = predictor.predict(
                point_coords=single_point,
                point_labels=point_label,
                multimask_output=False
            )

            # There may be multiple masks, but we assume one mask per point for simplicity
            mask = mask[0]  # Taking the first mask (index 0)
            
            # Integrate the mask with the combined mask
            combined_mask = combine_masks(combined_mask, mask)
    
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA out of memory for image {zslice_id}, switching to CPU.")
            device = torch.device('cpu')
            sam2_model.to(device)
            predictor = SAM2ImagePredictor(sam2_model)
            predictor.set_image(image)
            
            # Re-run the prediction on CPU
            mask, scores, _ = predictor.predict(
                point_coords=single_point,
                point_labels=point_label,
                multimask_output=False
            )
            mask = mask[0]
            combined_mask = combine_masks(combined_mask, mask)
        
    # Save the combined mask as an image
    mask_image_path = f"/home/codee/scratch/segment-anything-2/predictedpv/mask_{zslice_id}.png"
    print(f"Saving combined mask to: {mask_image_path}")
    save_mask_as_image(combined_mask, mask_image_path)
    '''
    # Visualization: Overlay the combined mask onto the original image
    input_label = [1] * len(points)
    show_masks(image, [combined_mask], [1.0], point_coords=points, input_labels=input_label)

    # Save the output visualization image
    output_path = f"/home/codee/scratch/segment-anything-2/predictedpv/pv_{zslice_id}.png"
    plt.savefig(output_path)
    plt.close()
    '''
    return mask_image_path

def main():
    set_start_method('spawn', force=True)
    # Select the device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directories for input images and SWC files
    input_em_dir = "/home/codee/scratch/segment-anything-2/test_human_liver"
    input_swc_file = "/home/codee/scratch/segment-anything-2/locationswc/portal_vein_3053886.swc"

    points_batch, img_batch, zslice_id_batch = get_batch(input_em_dir, input_swc_file)

    # Prepare the arguments for parallel processing
    tasks = [(image, points, zslice_id) for image, points, zslice_id in zip(img_batch, points_batch, zslice_id_batch)]

    # Use multiprocessing Pool to parallelize the tasks
    with Pool(processes=6) as pool:
        results = pool.map(process_single_image, tasks)

    print(f"Processing complete. Output saved to {results}")

if __name__ == "__main__":
    main()

    '''
    predictor.set_image(image)
    input_point = np.array([[900, 800]])
    input_label = np.array([1])
    plt.figure(figsize=(10, 10))
    origin_image = Image.open('/home/codee/scratch/datasets/human_liver_imageTs/human_100_0000.png')
    show_points(np.array([[9000, 8000]]), input_label, plt.gca())
    masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    show_masks(origin_image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
    '''