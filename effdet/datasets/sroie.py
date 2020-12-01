import os
from PIL import Image
import random

import numpy as np
import imgaug
import imgaug.augmentables as ia
import imgaug.augmenters as iaa
import torch
import torchvision as tv
from torch.utils.data import Dataset

from ..east_utils import get_input_image_and_bboxes, scale_bboxes, create_ground_truth, resizer


def decaying(start, stop, decay):
    """Yield an infinite series of linearly decaying values."""

    curr = float(start)
    while True:
        yield max(curr, stop)
        curr -= decay


def get_image(image_path):
    pil_image = Image.open(image_path).convert('RGB')
    np_image = np.array(pil_image)
    return np_image


def get_bboxes(annotations_path, shape):
    txt_annotations = open(annotations_path).read().split('\n')
    boxes = []
    for row in txt_annotations[:-1]:
        x0, y0, _, _, x2, y2, _, _, _ = row.split(',', 8)
        boxes.append(ia.BoundingBox(int(x0), int(y0), int(x2), int(y2)))
    boxes_on_image = ia.BoundingBoxesOnImage(boxes, shape=shape)
    return boxes_on_image


def get_image_and_bboxes(image_path):
    annotations_path = image_path.replace('.jpg', '.txt')
    image = get_image(image_path)
    bboxes = get_bboxes(annotations_path, shape=image.shape)
    return image, bboxes


class SROIEDataset(Dataset):

    def __init__(self, image_files, folderpath, height, width, scale,
                 ia_tfms=None,
                 do_gray=True,
                 test=False,
                 tfms_decay=(0.9, 0.0, 1e-5),
                 mean=torch.tensor([0.485, 0.456, 0.406]),
                 std=torch.tensor([0.229, 0.224, 0.225])):
        self.image_files = image_files
        self.folderpath = folderpath
        self.height = height
        self.width = width
        self.scale = scale
        self.mean = mean
        self.std = std
        self.test = test
        self.do_gray = do_gray
        self.gray_tfms = iaa.Grayscale(alpha=1.0)
        self.ia_tfms = ia_tfms
        self.decay = decaying(*tfms_decay)

        self.torch_tfms = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=self.mean, std=self.std)])

    def __len__(self,):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folderpath, self.image_files[idx])

        if self.test:
            image = get_image(image_path)
            if self.do_gray:
                image = self.gray_tfms(image=image)

            image = resizer(image=image, size=(self.height, self.width))
            return self.torch_tfms(image)
        else:
            image, bboxes = get_image_and_bboxes(image_path)

            size = (self.height, self.width)

            image, bboxes = get_input_image_and_bboxes(
                image, bboxes, image_size=size, scale=self.scale)

            if self.do_gray:
                image = self.gray_tfms(image=image)

            if self.ia_tfms is not None and random.random() < next(self.decay):
                image, bboxes = self.ia_tfms(
                    image=image, bounding_boxes=bboxes)
                bboxes = bboxes.clip_out_of_image()

            boxes, mask_boxes = scale_bboxes(bboxes, pct=0.7)

            gt_image, loss_mask = create_ground_truth(
                boxes, mask_boxes, size, self.scale)

            return self.torch_tfms(image), gt_image.astype(np.float32)
