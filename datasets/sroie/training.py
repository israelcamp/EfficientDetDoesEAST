# NOT CHANGED TO FIT NEW VERSION

from argparse import Namespace
import os

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import imgaug.augmentables as ia
import imgaug.augmenters as iaa

from effdet.east import EfficientDetDoesEAST
from effdet.east import EASTLoss
from effdet.east import decode

from .dataset import SROIEDataset


class LightningBase:

    '''
        hparams needs to contain:
            - lr (float)
            - optimizer (str)
            - optimizer_kwargs (Dict[str,Optional])
            - train_batch_size (int)
            - val_batch_size (int)
            - shuffle_train (bool)
            - num_workers (int)
        Properties needed:
            - train_dataset (Dataset)
            - val_dataset (Dataset)
            - test_dataset (Dataset)
    '''

    def _average_key(self, outputs, key):
        return torch.stack([o[key] for o in outputs]).float().mean()

    def get_dataloader(self, dataset, batch_size, shuffle, num_workers):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    def get_optimizer(self,):
        optimizer_name = self.hparams.optimizer
        lr = self.hparams.lr
        optimizer_hparams = self.hparams.optimizer_kwargs
        optimizer = getattr(torch.optim, optimizer_name)
        return optimizer(self.parameters(), lr=lr, **optimizer_hparams)

    def train_dataloader(self,):
        return self.get_dataloader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=self.hparams.shuffle_train,
            num_workers=self.hparams.num_workers
        )

    def val_dataloader(self,):
        return self.get_dataloader(
            self.valid_dataset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )

    def test_dataloader(self,):
        return self.get_dataloader(
            self.test_dataset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        return optimizer


class SROIEBase(LightningBase):

    '''
        hparams needs to contain:
            - test_pct (float)
            - val_pct (float)
            - width (int)
            - height (int)
            - scale (int)
            - do_gray (bool)
            - tfms_decay (Tuple[float, float, float])
        Properties needed:
            - folderpath (Path-like)
    '''

    def get_image_files_from_folder(self,):
        listedir = os.listdir(self.folderpath)
        image_files = [f for f in listedir if f.endswith(
            '.jpg') and f.replace('.jpg', '.txt') in listedir]
        image_files.sort()
        return image_files

    def get_ia_tfms(self,):
        return iaa.SomeOf(2, [
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-2, 2),
                cval=255,
            ),
            iaa.AdditiveGaussianNoise(scale=(0., 0.1*255)),
            iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
        ])

    def separate_images(self, image_files):
        n = len(image_files)
        test_size = round(self.hparams.test_pct * n)
        val_size = round(self.hparams.val_pct * n)
        train_files = image_files[test_size + val_size:]
        valid_files = image_files[test_size:test_size + val_size]
        test_files = image_files[:test_size]
        return train_files, valid_files, test_files

    def construct_dataset(self, image_files, add_tfms=True):
        return SROIEDataset(
            image_files=image_files,
            folderpath=self.folderpath,
            height=self.hparams.height,
            width=self.hparams.width,
            scale=self.hparams.scale,
            do_gray=self.hparams.do_gray,
            tfms_decay=self.hparams.tfms_decay,
            ia_tfms=self.get_ia_tfms() if add_tfms else None
        )

    def prepare_data(self,):
        image_files = self.get_image_files_from_folder()
        train_files, valid_files, test_files = self.separate_images(
            image_files)
        self.train_dataset = self.construct_dataset(train_files, True)
        self.valid_dataset = self.construct_dataset(valid_files, False)
        self.test_dataset = self.construct_dataset(test_files, False)


class EfficientDetOnPL(EfficientDetDoesEAST, pl.LightningModule):

    coef2size = {
        0: 512,
        1: 640,
        2: 768,
        3: 896,
        4: 1024,
        5: 1280,
        6: 1280,
        7: 1536
    }

    def _handle_batch(self, batch):
        image, gt = batch
        scores = self(image)
        loss = self.loss_fct(gt, scores)
        return (loss, scores)

    def _handle_eval_batch(self, batch):
        outputs = self._handle_batch(batch)
        return outputs

    def _handle_eval_epoch_end(self, outputs, phase):
        loss_avg = self._average_key(outputs, f'{phase}_loss')
        return loss_avg

    def get_loss_fct(self, lamb=1.):
        return EASTLoss(lamb=lamb)

    def training_step(self, batch, batch_idx):
        outputs = self._handle_batch(batch)
        return {'loss': outputs[0]}

    def validation_step(self, batch, batch_idx):
        outputs = self._handle_eval_batch(batch)
        return {'val_loss': outputs[0]}

    def test_step(self, batch, batch_idx):
        outputs = self._handle_eval_batch(batch)
        return {'test_loss': outputs[0]}

    def validation_epoch_end(self, outputs):
        loss_avg = self._handle_eval_epoch_end(outputs, phase='val')
        progress_bar = {'val_loss': loss_avg}
        return {'val_loss': loss_avg, 'progress_bar': progress_bar}

    def test_epoch_end(self, outputs):
        loss_avg = self._handle_eval_epoch_end(outputs, phase='test')
        return {'test_loss': loss_avg}

    def predict_one(self, score_maps, scale=4, threshold=0.6, nms_iou=0.01):
        '''
            Arguments:
                - score_maps (torch.Tensor of shape (5, H, W))
        '''
        np_boxes = decode(score_maps, scale=scale,
                          threshold=threshold, nms_iou=nms_iou)
        shape = (score_maps.shape[1], score_maps.shape[2], 3)
        bboxes = ia.BoundingBoxesOnImage.from_xyxy_array(np_boxes, shape)
        return np_boxes, bboxes


class EfficientDetSROIETuner(SROIEBase, EfficientDetOnPL):

    default_hparams = {
        "test_pct": 0.05,
        "val_pct": 0.1,
        "lr": 5e-4,
        "optimizer": 'Adam',
        "optimizer_kwargs": {},
        "scale": 4,
        "train_batch_size": 2,
        "val_batch_size": 2,
        "shuffle_train": True,
        "num_workers": 4,
        "lamb": 1.,
        "deterministic": False,
        "seed": 1,
        "do_gray": True,
        "tfms_decay": (0.9, 0.0, 1e-5)
    }

    def __init__(self, folderpath, compound_coef=0, hparams=None):
        super(EfficientDetOnPL, self).__init__(
            compound_coef=compound_coef, load_weights=False)

        self.folderpath = folderpath
        self.compound_coef = compound_coef
        self.hparams = self._construct_hparams(hparams)

        self.compound_coef = compound_coef
        self.loss_fct = self.get_loss_fct(lamb=self.hparams.lamb)

    def _construct_hparams(self, hparams):
        default_hparams = self.default_hparams.copy()
        if hparams is not None:
            default_hparams.update(hparams)
        default_hparams['compound_coef'] = self.compound_coef
        default_hparams['folderpath'] = self.folderpath

        if 'height' not in self.default_hparams:
            default_hparams['height'] = self.coef2size[self.compound_coef]
        if 'width' not in self.default_hparams:
            default_hparams['width'] = default_hparams['height'] // 2

        if default_hparams['deterministic']:
            pl.seed_everything(default_hparams['seed'])

        return Namespace(**default_hparams)
