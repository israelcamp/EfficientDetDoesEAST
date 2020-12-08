import math

import imgaug.augmenters as iaa
import numpy as np
import torch
import torchvision as tv


def resizer(image=None, bboxes=None, size=(512, 512)):
    kwargs = {}
    if image is not None:
        kwargs['image'] = image
    if bboxes is not None:
        kwargs['bounding_boxes'] = bboxes
    assert len(kwargs), 'Needs image or bboxes to resize'
    return iaa.Resize({'width': size[1], 'height': size[0]})(**kwargs)


def get_input_image_and_bboxes(image, bboxes, image_size, scale):
    h, w = image_size
    mh, mw = h // scale, w // scale
    image = resizer(image=image, size=(h, w))
    bboxes = resizer(bboxes=bboxes, size=(mh, mw))
    return image, bboxes


def scale_bboxes(bboxes, pct=0.7, minh=2, minw=4):
    # original without removed pixels
    boxes = bboxes.to_xyxy_array(np.int32)
    # remove pixels from boxes
    mask_boxes = np.array(
        [scale_box(box, pct=pct, minh=minh, minw=minw) for box in boxes])
    return boxes, mask_boxes


def scale_box(box, pct=0.7, minh=2, minw=4):
    '''
        Parameters:
            - box (np.array or list-like): shape (N, 4) (x1, y1, x2, y2) for each box
            - pct (int): scale factor, (1. - pct) pixels will be removed from each box
            - minh (int): minimum height of the box
            - minw (int): minimum width of the box
    '''
    x1, y1, x2, y2 = box
    height = y2 - y1
    width = x2 - x1

    remove_y = math.floor(height - height * (1. - pct))
    remove_x = math.floor(width - width * (1. - pct))

    center_y = int((y2 + y1) // 2)
    center_x = int((x2 + x1) // 2)

    dx = max(minw // 2, math.floor(remove_x / 2))
    dy = max(minh // 2, math.floor(remove_y / 2))

    x1 = max(x1, center_x - dx)
    x2 = min(x2, center_x + dx)
    y1 = max(y1, center_y - dy)
    y2 = min(y2, center_y + dy)
    return x1, y1, x2, y2


def create_ground_truth(boxes, mask_boxes, size, scale):
    h, w = size
    new_height = h // scale
    new_width = w // scale
    gt_image = np.zeros((5, new_height, new_width))
    y_loss_mask = np.zeros((new_height, new_width))

    for bbox, mbox in zip(boxes, mask_boxes):
        x1, y1, x2, y2 = mbox.astype(np.int32)
        ox1, oy1, ox2, oy2 = bbox.astype(np.int32)

        y_loss_mask[oy1:oy2, ox1:ox2] = 1.

        for dx in range(x1, x2):
            for dy in range(y1, y2):
                assert dy - oy1 >= 0
                gt_image[0, dy, dx] = 1  # score
                gt_image[1, dy, dx] = (dy - oy1) * scale  # top
                gt_image[2, dy, dx] = (oy2 - dy) * scale  # bottom
                gt_image[3, dy, dx] = (dx - ox1) * scale  # left
                gt_image[4, dy, dx] = (ox2 - dx) * scale  # right

    y_loss_mask = 1. * (y_loss_mask == gt_image[0])
    return gt_image, y_loss_mask


def decode(pred_image, scale, threshold=0.8, nms_iou=0.01):
    probas = []
    boxes = []
    nz_coords = torch.nonzero(pred_image[0] > threshold)
    for t in zip(nz_coords):
        y, x = t[0]
        proba = pred_image[0, y, x]
        probas.append(proba)
        d1, d2, d3, d4 = pred_image[1:5, y, x]
        box = [scale * x - d3, scale * y - d1,
               scale * x + d4, scale * y + d2]
        boxes.append(box)
    probs, boxes = torch.tensor(probas), torch.tensor(boxes)

    if len(boxes.shape) > 1:
        keep_indices = tv.ops.nms(boxes, probs, nms_iou)
        keep_boxes = boxes[keep_indices]
        return keep_boxes
    return None
