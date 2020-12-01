import torch
from torch import nn


class BalancedBCE(nn.Module):

    def forward(self, y_true, y_pred, loss_mask=None):
        bs = y_true.shape[0]
        size = y_true.shape[-1] * y_true.shape[-2]

        y_true = y_true.view(bs, -1)
        y_pred = y_pred.view(bs, -1)

        beta = 1. - y_true.sum(-1, keepdim=True) / size
        first_term = beta * y_true * torch.log(y_pred)
        second_term = (1. - beta) * (1. - y_true) * torch.log(1. - y_pred)
        loss = - first_term - second_term

        if loss_mask is not None:
            loss_mask = loss_mask.view(bs, -1)
            loss = loss * loss_mask

        return loss.mean()


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


class EASTLoss(nn.Module):

    def __init__(self, lamb=1.):
        super().__init__()
        self.lamb = lamb
        self.bce = BalancedBCE()
        self.iou = IoULoss()

    def forward(self, y_true, y_pred, loss_mask=None):
        bce = self.bce(y_true[:, 0], y_pred[:, 0], loss_mask)
        iou = self.iou(y_pred[:, 1:], y_true[:, 1:], y_true[:, 0])
        return bce + self.lamb * iou
