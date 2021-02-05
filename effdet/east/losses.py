import torch
from torch import nn

from kornia.losses import FocalLoss, DiceLoss as KorniaDiceLoss

class DiceLoss(nn.Module):
    
    def forward(self, y_true, scores, eps=1e-8):
        # compute the actual dice score
        dims = (1, 2)
        intersection = torch.sum(scores * y_true, dims)
        cardinality = torch.sum(scores + y_true, dims)

        dice_score = 2. * intersection / (cardinality + eps)
        return torch.mean(-dice_score + 1.)


class BalancedBCE(nn.Module):

    def forward(self, y_true, y_pred, loss_mask=None):
        bs = y_true.shape[0]
        size = y_true.shape[-1] * y_true.shape[-2]

        y_true = y_true.view(bs, -1)
        y_pred = y_pred.view(bs, -1)

        beta = 1. - y_true.sum(-1, keepdim=True) // size
        first_term = beta * y_true * torch.log(y_pred)
        second_term = (1. - beta) * (1. - y_true) * torch.log(1. - y_pred)
        loss = - first_term - second_term

        if loss_mask is not None:
            loss_mask = loss_mask.view(bs, -1)
            loss = loss * loss_mask
        return loss.mean()

        # n = y_true.sum()
        # N = y_true.numel()
        # beta = 1.0 - 1.0 * n / N
        # loss = -beta * y_true * torch.log(y_pred) - (1 - beta) * (1 - y_true) * torch.log(1 - y_pred)
        # return loss.sum()



class IoULoss(nn.Module):

    @staticmethod
    def bbox_loss(y_pred, y_true, y_true_score):
        assert y_true.min() >= 0, y_true.min().item()

        d1_true, d2_true, d3_true, d4_true = torch.split(y_true, 1, 1)
        d1_pred, d2_pred, d3_pred, d4_pred = torch.split(y_pred, 1, 1)
        area_true = (d1_true + d3_true) * (d2_true + d4_true)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_intersect = torch.min(d2_true, d2_pred) + torch.min(d4_true, d4_pred)
        h_intersect = torch.min(d1_true, d1_pred) + torch.min(d3_true, d3_pred)
        area_intersect = w_intersect * h_intersect
        area_union = area_true + area_pred - area_intersect
        aabb_loss = -torch.log((area_intersect + 1.0)/(area_union + 1.0))
        cbox_loss = torch.sum(aabb_loss * y_true_score) / y_true_score.sum()
        return cbox_loss

    def forward(self, y_pred, y_true, y_true_score):
        return self.bbox_loss(y_pred, y_true, y_true_score)


class L1(nn.Module):

    def __init__(self,):
        super().__init__()
        self.loss_fct = nn.SmoothL1Loss()

    def smooth_l1(self, y_pred, y_true, y_true_score):
        # repeat the channels
        y_true_score = torch.stack(4*[y_true_score], dim=1)
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        y_true_score = y_true_score.reshape(-1)

        # filter the ones that should learn
        y_pred_active = y_pred[y_true_score == 1]
        y_true_active = y_true[y_true_score == 1]
        loss = self.loss_fct(y_pred_active.unsqueeze(1),
                             y_true_active.unsqueeze(1))
        return loss

    def forward(self, y_pred, y_true, y_true_score):
        return self.smooth_l1(y_pred, y_true, y_true_score)

class EASTLoss(nn.Module):

    def __init__(self, 
                iou_weight: float = 1., 
                dice_weight: float = 1.,
                bce_weight: float = 1.,
                l1_weight: float = 1.,
                do_l1: bool = True,
                do_bce: bool = False, 
                do_dice: bool = True):
        super().__init__()
        self.iou_weight = iou_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.do_l1 = do_l1
        self.l1_weight = l1_weight
        self.dice = DiceLoss()
        self.bce = BalancedBCE()
        self.iou = IoULoss()
        self.l1 = L1()

        self.do_bce = do_bce
        self.do_dice = do_dice

    def forward(self, y_true, y_pred, loss_mask=None):
        losses = {}

        # IoU
        iou = self.iou(y_pred[:, 1:], y_true[:, 1:], y_true[:, 0])
        losses['iou'] = self.iou_weight * iou

        # Dice
        if self.do_l1:
            l1 = self.l1(y_pred[:, 1:], y_true[:, 1:], y_true[:, 0].long())
            losses['l1'] = self.l1_weight * l1

        # Dice
        if self.do_dice:
            dice = self.dice(y_true[:, 0], y_pred[:, 0])
            losses['dice'] = self.dice_weight * dice

        # BCE
        if self.do_bce:
            bce = self.bce(y_true[:, 0], y_pred[:, 0])
            losses['bce'] = self.bce_weight * bce

        loss = sum(losses.values())

        outputs = (loss, losses)
        return outputs

class MultiLabelEASTLoss(nn.Module):

    def __init__(self, 
                iou_weight: float = 1., 
                l1_weight: float = 1.,
                dice_weight: float = 1.,
                focal_weight: float= 1.,
                ce_weight: float = 1.,
                do_iou: bool = True,
                do_focal: bool = True,
                do_l1: bool = True,
                do_dice: bool = False,
                do_ce: bool = False,
                focal_alpha: float = 0.25,
                focal_reduction: str = 'mean'):
        super().__init__()

        self.do_iou = do_iou
        self.iou_weight = iou_weight
        self.iou = IoULoss()
        
        self.do_l1 = do_l1
        self.l1_weight = l1_weight
        self.l1 = L1()

        self.do_dice = do_dice
        self.dice_weight = dice_weight
        self.dice = KorniaDiceLoss()

        self.do_focal = do_focal
        self.focal_weight = focal_weight
        self.focal = FocalLoss(alpha=focal_alpha, reduction=focal_reduction)

        self.do_ce = do_ce
        self.ce_weight = ce_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, y_true, y_pred, loss_mask=None):
        losses = {}

        # Dice
        if self.do_dice:
            dice = self.dice(y_pred[:, :-4], y_true[:, 0].long())
            losses['dice'] = self.dice_weight * dice
            
        # Focal
        if self.do_focal:
            focal = self.focal(y_pred[:, :-4], y_true[:, 0].long())
            losses['focal'] = self.focal_weight * focal

        if self.do_ce:
            ce = self.ce(y_pred[:, :-4], y_true[:, 0].long())
            losses['ce'] = self.ce_weight * ce
            
        binary_y_true = 1 * (y_true[:, 0] > 0)
        # IoU
        if self.do_iou:
            iou = self.iou(y_pred[:, -4:], y_true[:, 1:], binary_y_true)
            losses['iou'] = self.iou_weight * iou

        # Dice
        if self.do_l1:
            l1 = self.l1(y_pred[:, -4:], y_true[:, 1:], binary_y_true.long())
            losses['l1'] = self.l1_weight * l1

        loss = sum(losses.values())

        outputs = (loss, losses)
        return outputs