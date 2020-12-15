from pathlib import Path
import json
import datetime
import copy
import os

# Third Parties
import numpy as np
from PIL import Image
import imgaug.augmentables as ia
import imgaug.augmenters as iaa

# Torch
from torch.utils.data import Dataset
import torchvision as tv

# Types
from typing import List, Tuple, Optional, Union
import dataclasses
from pydantic.dataclasses import dataclass
from pydantic import validate_arguments, validator, Field
from pydantic import DirectoryPath, FilePath

from effdet.east import get_input_image_and_bboxes, scale_bboxes, create_ground_truth

@dataclass
class COCO_Text:
    
    annotation_file: FilePath = Field(..., description="location of annotation file")
    
    def __post_init_post_parse__(self,):
        """
        Constructor of COCO-Text helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :return:
        """
        # load dataset
        self.dataset = {}
        self.anns = {}
        self.imgToAnns = {}
        self.catToImgs = {}
        self.imgs = {}
        self.cats = {}
        self.val = []
        self.test = []
        self.train = []
        if not self.annotation_file == None:
            assert os.path.isfile(self.annotation_file), "file does not exist"
            print('loading annotations into memory...')
            time_t = datetime.datetime.utcnow()
            dataset = json.load(open(self.annotation_file, 'r'))
            print(datetime.datetime.utcnow() - time_t)
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        self.imgToAnns = {int(cocoid): self.dataset['imgToAnns'][cocoid] for cocoid in self.dataset['imgToAnns']}
        self.imgs      = {int(cocoid): self.dataset['imgs'][cocoid] for cocoid in self.dataset['imgs']}
        self.anns      = {int(annid): self.dataset['anns'][annid] for annid in self.dataset['anns']}
        self.cats      = self.dataset['cats']
        self.val       = [int(cocoid) for cocoid in self.dataset['imgs'] if self.dataset['imgs'][cocoid]['set'] == 'val']
        self.test      = [int(cocoid) for cocoid in self.dataset['imgs'] if self.dataset['imgs'][cocoid]['set'] == 'test']
        self.train     = [int(cocoid) for cocoid in self.dataset['imgs'] if self.dataset['imgs'][cocoid]['set'] == 'train']
        print('index created!')

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('%s: %s'%(key, value))

    def filtering(self, filterDict, criteria):
        return [key for key in filterDict if all(criterion(filterDict[key]) for criterion in criteria)]

    def getAnnByCat(self, properties):
        """
        Get ann ids that satisfy given properties
        :param properties (list of tuples of the form [(category type, category)] e.g., [('readability','readable')] 
            : get anns for given categories - anns have to satisfy all given property tuples
        :return: ids (int array)       : integer array of ann ids
        """
        return self.filtering(self.anns, [lambda d, x=a, y=b:d[x] == y for (a,b) in properties])

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[]):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (list of tuples of the form [(category type, category)] e.g., [('readability','readable')] 
                : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = list(self.anns.keys())
        else:
            if not len(imgIds) == 0:
                anns = sum([self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns],[])
            else:
                anns = list(self.anns.keys())
            anns = anns if len(catIds)  == 0 else list(set(anns).intersection(set(self.getAnnByCat(catIds)))) 
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if self.anns[ann]['area'] > areaRng[0] and self.anns[ann]['area'] < areaRng[1]]
        return anns

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = list(self.imgs.keys())
        else:
            ids = set(imgIds)
            if not len(catIds) == 0:
                ids  = ids.intersection(set([self.anns[annid]['image_id'] for annid in self.getAnnByCat(catIds)]))
        return list(ids)

    def loadAnn(self, img_id: int):
        """
        Load anns with the specified ids.
        :param id (int)       : integer id specifying ann
        :return: anns (object) : loaded ann object
        """
        ids = self.getAnnIds(img_id)
        return [self.anns[id] for id in ids]

    def loadImg(self, id: int):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        return self.imgs[id]


class DatasetConfig:
    arbitrary_types_allowed = True

@dataclass(config=DatasetConfig)
class COCOTextDataset(Dataset):
    
    img_ids: List[int] =         Field(..., description="Image ids from COCO Dataset")
    img_dir: DirectoryPath =     Field(..., description="Path to dir with images")
    ct: COCO_Text =              Field(..., description="COCO Text instance")
    image_size:Tuple[int, int] = Field(default=(768, 768), description="Image size in (h,w)")
    scale: int =                 Field(default=4, description="Mask relative scale to image (h/scale, w/scale)")
    transforms: Optional[iaa.Sequential] = dataclasses.field(default=None, metadata="Albumentations transforms")
        
    to_torch = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: 2 * x - 1)
    ])


    def __len__(self,):
        return len(self.img_ids)
    
    
    def _img_and_annotations(self, idx):
        img = self.ct.loadImg(self.img_ids[idx])
        ann = self.ct.loadAnn(img['id'])
        
        path = os.path.join(self.img_dir, img['file_name'])
        image = np.array(Image.open(path).convert('RGB'))
        return image, ann
    
    def _make_bboxes_from_annotations(self, anns, shape):
        bboxes = []
        for a in anns:
            x0, y0, w, h = a['bbox']
            y1, x1 = y0 + h, x0 + w
            bboxes.append(ia.BoundingBox(x0, y0, x1, y1))
        bboxes = ia.BoundingBoxesOnImage(bboxes, shape=shape)
        return bboxes
    
    def __getitem__(self, idx):
        image, anns = self._img_and_annotations(idx)
        bboxes = self._make_bboxes_from_annotations(anns, image.shape)
        
        # HERE COMES THE AUGMENTATIONS
        if self.transforms is not None:
            image, bboxes = self.transforms(image=image, bounding_boxes=bboxes)
            bboxes = bboxes.remove_out_of_image_fraction(fraction=0.75)
            bboxes = bboxes.clip_out_of_image()

        # SCALES BOXES AND GENERATE GT
        image, bboxes = get_input_image_and_bboxes(image, bboxes, self.image_size, self.scale)
        bboxes, mask_boxes = scale_bboxes(bboxes)
        gt_image, _ = create_ground_truth(bboxes, mask_boxes, self.image_size, self.scale)
        
        image = self.to_torch(image)
        
        return image, gt_image.astype(np.float32)