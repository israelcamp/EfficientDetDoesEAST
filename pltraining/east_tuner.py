import argparse

import imgaug.augmentables as ia

import torchvision as tv
import torch

import pytorch_lightning as pl

# Mine
from effdet.east import EfficientDetDoesEAST, decode, resizer, EASTLoss, MultiLabelEAST
from effdet.east.losses import MultiLabelEASTLoss


def eastmask_to_image(image, emask, scale=4, ths=0.5, nms=0.01):
    xyxy = decode(emask, scale=scale, threshold=ths, nms_iou=nms)
    image = (image + 1)/2.
    if xyxy is not None:
        # bboxes and resize
        image = image.permute(1, 2, 0)
        bboxes = ia.BoundingBoxesOnImage.from_xyxy_array(
            xyxy.cpu().numpy(), shape=image.shape)
        bboxes = resizer(bboxes=bboxes, size=image.shape[:-1])
        image = tv.transforms.ToTensor()(
            bboxes.draw_on_image(image, color=(0, 255, 0), size=10))
    return image


def grid_from_batch(images, emasks, scale=4, ths=0.5, nms=0.01):
    imgs = []
    for img, em in zip(images, emasks):
        imgd = eastmask_to_image(img, em, scale=scale, ths=ths, nms=nms)
        imgs.append(imgd)
    grid = torch.stack(imgs)
    grid = tv.utils.make_grid(grid)
    return grid


class EfficientDetDoesPL(pl.LightningModule):

    '''
        Properties needed:
            - loss_fct (nn.Module)
    '''

    def get_optimizer(self,) -> torch.optim.Optimizer:
        optimizer_name = self.hparams.optimizer
        lr = self.hparams.lr
        optimizer_hparams = self.hparams.optimizer_kwargs
        optimizer = getattr(torch.optim, optimizer_name)
        return optimizer(self.parameters(), lr=lr, **optimizer_hparams)

    def _average_key(self, outputs, key: str) -> torch.FloatTensor:
        return torch.stack([o[key] for o in outputs]).float().mean()

    def _concat_lists_by_key(self, outputs, key):
        return sum([o[key] for o in outputs], [])

    def _handle_batch(self, batch):
        image, gt = batch
        scores = self(image)

        loss, losses = self.loss_fct(gt, scores)
        return (loss, scores, losses)

    def _handle_eval_batch(self, batch):
        outputs = self._handle_batch(batch)
        return outputs

    def _handle_eval_epoch_end(self, outputs, phase):
        loss_avg = self._average_key(outputs, f'{phase}_loss')
        return loss_avg

    ## FUNCTIONS NEEDED BY PYTORCH LIGHTNING ##

    def training_step(self, batch, batch_idx):
        if self.hparams.freeze_backbone_bn:
            self.model.backbone.freeze_bn()

        outputs = self._handle_batch(batch)
        loss = outputs[0]
        losses = outputs[-1]
        self.log('train_loss', loss, on_step=True, prog_bar=False, logger=True)
        try:
            self.logger.experiment.log_metric('nep_train_loss', loss)
            for key, value in losses.items():
                self.logger.experiment.log_metric(
                    f'nep_train_{key}_loss', value)
        except:
            pass
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, scores, _ = self._handle_eval_batch(batch)
        if batch_idx == self.hparams.val_batch_idx:
            try:
                grid = grid_from_batch(
                    batch[0].cpu(), scores.cpu(), scale=self.hparams.scale)
                self.logger.experiment.log_image(
                    'val_image', tv.transforms.ToPILImage()(grid))
            except:
                pass
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        loss = self._handle_eval_batch(batch)[0]
        return {'test_loss': loss}

    def validation_epoch_end(self, outputs):
        loss_avg = self._handle_eval_epoch_end(outputs, phase='val')
        self.log('val_loss', loss_avg, on_epoch=True,
                 prog_bar=True, logger=True)

    def test_epoch_end(self, outputs):
        loss_avg = self._handle_eval_epoch_end(outputs, phase='test')
        self.log('test_loss', loss_avg, on_epoch=True,
                 prog_bar=True, logger=True)
        return {'test_loss': loss_avg}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.get_optimizer()
        return optimizer

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class EASTUner(EfficientDetDoesPL):

    default_hparams = {
        "coef": 4,
        "repeat_bifpn": 3,
        "bifpn_channels": 128,
        "advprop": True,
        "factor2": False,
        "expand_bifpn": False,
        "freeze_backbone_bn": True,
        "lr": 5e-4,
        "optimizer": 'Adam',
        "optimizer_kwargs": {},
        "loss_hparams": {},
        "deterministic": False,
        "seed": 0,
        "val_batch_idx": 1
    }

    def __init__(self, hparams=None, **kwargs):

        self.model_kwargs = kwargs
        self.hparams = self._construct_hparams(hparams)

        super(EfficientDetDoesPL, self).__init__()

        self.model = EfficientDetDoesEAST(advprop=self.hparams.advprop,
                                          compound_coef=self.hparams.coef,
                                          expand_bifpn=self.hparams.expand_bifpn,
                                          factor2=self.hparams.factor2,
                                          repeat_bifpn=self.hparams.repeat_bifpn,
                                          bifpn_channels=self.hparams.bifpn_channels
                                          )

        self.loss_fct = self.get_loss_fct()

    def get_loss_fct(self,):
        return EASTLoss(**self.hparams.loss_hparams)

    def _construct_hparams(self, hparams):
        default_hparams = self.default_hparams.copy()

        scale = 4  # default
        if default_hparams['expand_bifpn']:
            scale = scale // 2
        if default_hparams['factor2']:
            scale = scale // 2
        default_hparams['scale'] = scale

        if hparams is not None:
            default_hparams.update(hparams)

        default_hparams.update(self.model_kwargs)

        if default_hparams['deterministic']:
            pl.seed_everything(default_hparams['seed'])

        return argparse.Namespace(**default_hparams)


def decode_multi(pred_image, scale, threshold=0.8, nms_iou=0.01, nms_function=tv.ops.nms):
    probas = []
    boxes = []
    
    segmaps = pred_image[:-4]
    pred_labels = segmaps.argmax(dim=0)
    probs = segmaps.softmax(dim=0)
    probs, _ = probs.max(dim=0)
    
    nz_coords = torch.nonzero(pred_labels)
    
    for t in zip(nz_coords):
        y, x = t[0]
        proba = probs[y, x]
        probas.append(proba)
        d1, d2, d3, d4 = pred_image[-4:, y, x]
        box = [scale * x - d3, scale * y - d1,
               scale * x + d4, scale * y + d2]
        boxes.append(box)
    probs, boxes = torch.tensor(probas), torch.tensor(boxes)

    if len(boxes.shape) > 1:
        keep_indices = nms_function(boxes, probs, nms_iou)
        keep_boxes = boxes[keep_indices]
        return keep_boxes
    return None

def eastmask_to_image_multi(image, emask, scale=4, ths=0.5, nms=0.01):
    xyxy = decode_multi(emask, scale=scale, threshold=ths, nms_iou=nms)
    image = (image + 1)/2.
    if xyxy is not None:
        # bboxes and resize
        image = image.permute(1, 2, 0)
        bboxes = ia.BoundingBoxesOnImage.from_xyxy_array(
            xyxy.cpu().numpy(), shape=image.shape)
        bboxes = resizer(bboxes=bboxes, size=image.shape[:-1])
        image = tv.transforms.ToTensor()(
            bboxes.draw_on_image(image, color=(0, 255, 0), size=10))
    return image


def grid_from_batch_multi(images, emasks, scale=4, ths=0.5, nms=0.01):
    imgs = []
    for img, em in zip(images, emasks):
        imgd = eastmask_to_image_multi(img, em, scale=scale, ths=ths, nms=nms)
        imgs.append(imgd)
    grid = torch.stack(imgs)
    grid = tv.utils.make_grid(grid)
    return grid

class MultiLabelEASTUner(EASTUner):

    default_hparams = {
        "coef": 4,
        "repeat_bifpn": 3,
        "bifpn_channels": 128,
        "advprop": True,
        "factor2": False,
        "expand_bifpn": False,
        "freeze_backbone_bn": True,
        "lr": 5e-4,
        "optimizer": 'Adam',
        "optimizer_kwargs": {},
        "loss_hparams": {},
        "deterministic": False,
        "seed": 0,
        "val_batch_idx": 1,
        "num_labels": 2
    }

    def __init__(self, hparams=None, **kwargs):

        self.model_kwargs = kwargs
        self.hparams = self._construct_hparams(hparams)

        super(EASTUner, self).__init__()

        self.model = MultiLabelEAST(advprop=self.hparams.advprop,
                                    compound_coef=self.hparams.coef,
                                    expand_bifpn=self.hparams.expand_bifpn,
                                    factor2=self.hparams.factor2,
                                    repeat_bifpn=self.hparams.repeat_bifpn,
                                    bifpn_channels=self.hparams.bifpn_channels,
                                    num_labels=self.hparams.num_labels,
                                    geo_layers=self.hparams.geo_layers,
                                    class_layers=self.hparams.class_layers
                                    )

        self.loss_fct = self.get_loss_fct()


    def get_loss_fct(self,):
        return MultiLabelEASTLoss(**self.hparams.loss_hparams)

    def validation_step(self, batch, batch_idx):
        loss, scores, _ = self._handle_eval_batch(batch)
        if batch_idx == self.hparams.val_batch_idx:
            try:
                grid = grid_from_batch_multi(
                    batch[0].cpu(), scores.cpu(), scale=self.hparams.scale, nms=1e-6)
                self.logger.experiment.log_image(
                    'val_image', tv.transforms.ToPILImage()(grid))
            except:
                pass
        return {'val_loss': loss}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.get_optimizer()
        return optimizer