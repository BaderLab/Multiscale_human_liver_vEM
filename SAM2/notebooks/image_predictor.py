import sys
sys.path.append('/home/codee/scratch/segment-anything-2')
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
    print(pos_points)
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
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
        plt.axis('off')
        plt.savefig(f"/home/codee/scratch/segment-anything-2/notebooks/images/predicted{i}.png")
        plt.show()

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
        print(slicenumber(image_file))
        zslice_id_batch.append(slicenumber(image_file))

    return points_batch, images_batch, zslice_id_batch

def main():
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    np.random.seed(3)

    # Initialize SAM2 model
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir="/home/codee/scratch/segment-anything-2/sam2_configs", version_base='1.2')
    sam2_checkpoint = "/home/codee/scratch/segment-anything-2/checkpoints/sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    predictor = SAM2ImagePredictor(sam2_model)

    # Directories for input images and SWC files
    input_em_dir = "/home/codee/scratch/segment-anything-2/test_human_liver"
    input_swc_file = "/home/codee/scratch/segment-anything-2/locationswc/portal_vein_3053886.swc"

    points_batch, img_batch, zslice_id_batch = get_batch(input_em_dir, input_swc_file)

    # Assume points_labels_batch is same size as points_batch and consists of all 1s
    points_labels_batch = [[1] * len(points) for points in points_batch]

    predictor.set_image_batch(img_batch)

    masks_batch, scores_batch, _ = predictor.predict_batch(
        points_batch,
        points_labels_batch,
        box_batch=None,
        multimask_output=False
    )
    print(len(masks_batch))
    for image, points, mask, scores, zslice_id in zip(img_batch, points_batch, masks_batch, scores_batch, zslice_id_batch):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        input_label = [1] * len(points)
        #for mask in masks:
        show_masks(image, mask, scores, point_coords=points, input_labels=input_label)

        plt.savefig(f"/home/codee/scratch/segment-anything-2/predictedpv/atest_{zslice_id}.png")

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