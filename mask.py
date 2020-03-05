import cv2
import numpy as np
import os
import sys
from samples import coco
from mrcnn import utils
from mrcnn import model as modellib

original_image = 'face.jpg'
image = cv2.imread(original_image)
cv2.imwrite('original_image.jpg', image)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image.jpg', image)

COCO_MODEL_PATH = 'mask_rcnn_coco.h5'

# Change the config infermation
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
config = InferenceConfig()

model = modellib.MaskRCNN(
    mode="inference", model_dir='mask_rcnn_coco.h5', config=config
)
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = [
    'BG', 'person']

def apply_mask(image, mask):
    image[:, :, 0] = np.where(
        mask == 0,
        gray_image[:, :],
        image[:, :, 0]
    )
    image[:, :, 1] = np.where(
        mask == 0,
        gray_image[:, :],
        image[:, :, 1]
    )
    image[:, :, 2] = np.where(
        mask == 0,
        gray_image[:, :],
        image[:, :, 2]
    )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    # max_area will save the largest object for all the detection results
    max_area = 0
    
    # n_instances saves the amount of all objects
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        # compute the square of each object
        y1, x1, y2, x2 = boxes[i]
        square = (y2 - y1) * (x2 - x1)

        # use label to select person object from all the 80 classes in COCO dataset
        label = names[ids[i]]
        if label == 'person':
            # save the largest object in the image as main character
            # other people will be regarded as background
            if square > max_area:
                max_area = square
                mask = masks[:, :, i]
            else:
                continue
        else:
            continue

        # apply mask for the image
    # by mistake you put apply_mask inside for loop or you can write continue in if also
    image = apply_mask(image, mask)
        
    return image



results = model.detect([image], verbose=0)
r = results[0]
frame = display_instances(
    image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
)
cv2.imshow('save_image', frame)

# Wait for keys to exit or save
k = cv2.waitKey(0)
if k == 27:                 
    cv2.destroyAllWindows()
elif k == ord('s'):        
    cv2.imwrite('save_image.jpg', image)
    cv2.destroyAllWindows()

