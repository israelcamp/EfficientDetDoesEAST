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


def decode(pred_image, scale, threshold=0.8, nms_iou=0.01, nms_function=tv.ops.nms):
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
        keep_indices = nms_function(boxes, probs, nms_iou)
        keep_boxes = boxes[keep_indices]
        return keep_boxes
    return None


def intersection(g, p):
    return tv.ops.box_iou(g.view(1,4), p.view(1,4)).view(-1).item()

def weighted_merge(g, p, ths=0.5):
    n = 4
    if p[n] >  ths:
        g[:n] = (g[n] * g[:n] + p[n] * p[:n]) / (g[n] + p[n])
        g[n] = max(g[n], p[n])
    return g

def nms_locality(polys, thres=0.2):
    ''' Locality aware nms for EAST
    :param polys: a [N, 4] numpy array. First 8 coordinates, then score.
    :return: polys after nms
    '''
    S = []
    p = None
    for g in polys:
        if p is not None and intersection(g[:4], p[:4]) > thres:
            p = weighted_merge(g, p)
        else:
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)
    return S

def decode_multi(pred_image, scale, nms_iou=0.01, standard_nms=True):
    probas = []
    boxes = []
    labels = []
    
    segmaps = pred_image[:-4]
    
    pred_labels = segmaps.argmax(dim=0)
    probs = segmaps.softmax(dim=0)
    probs, _ = probs.max(dim=0)
    
    nz_coords = torch.nonzero(pred_labels)
    
    for t in zip(nz_coords):
        y, x = t[0]
        proba = probs[y, x]
        l = pred_labels[y, x]
        
        probas.append(proba)
        labels.append(l)
        
        d1, d2, d3, d4 = pred_image[-4:, y, x]
        box = [scale * x - d3, scale * y - d1,
               scale * x + d4, scale * y + d2]
        boxes.append(box)
    probs, boxes, labels = torch.tensor(probas), torch.tensor(boxes), torch.tensor(labels)

    if len(boxes.shape) > 1:
        if standard_nms:
            keep_indices = tv.ops.nms(boxes, probs, nms_iou)
            
            keep_boxes = boxes[keep_indices]
            keep_labels = labels[keep_indices]
                    
            return keep_boxes, keep_labels
    
        boxes2label = {}
        
        for l, b, p in zip(labels, boxes, probs):
            l = l.item()
            
            if l not in boxes2label:
                boxes2label[l] = []
            
            boxes2label[l].append(torch.cat([b.view(1,4),p.view(1,1)], dim=1))
                        
        boxes2label = {
            l:nms_locality(torch.cat(list(v), dim=0), thres=0.5) for l, v in boxes2label.items()
        }
        
        matrices = []
        for l, v in boxes2label.items():
            m = torch.stack(v)
            m = torch.cat([m, l * torch.ones(len(m), 1)], dim=1)
            matrices.append(m)
            
        matrix = torch.cat(matrices, dim=0)
            
        boxes, probs, labels = matrix[:,:4], matrix[:, 4], matrix[:, 5].long()
        
        keep_indices = tv.ops.nms(boxes, probs, nms_iou)
        
        keep_boxes = boxes[keep_indices]
        keep_labels = labels[keep_indices]
                
        return keep_boxes, keep_labels
    return None