# Author: israel & dscarmo
# Modified From: Zylo11

import torch
from torch import nn

from ..segmentation import EfficientDetForSemanticSegmentation
from ..base.model import SegmentationClasssificationHead

class EfficientDetDoesEAST(nn.Module):

    def __init__(self,
                 advprop: bool = True,
                 compound_coef: int = 4,
                 expand_bifpn: bool = False,
                 n_seg_channels: int = 40,
                 repeat_bifpn: int = 3,
                 bifpn_channels: int = 128,
                 factor2: bool = False):
        super().__init__()
        self.num_classes = n_seg_channels
        self.backbone = EfficientDetForSemanticSegmentation(advprop=advprop,
                                                            num_classes=self.num_classes,
                                                            apply_sigmoid=False,
                                                            compound_coef=compound_coef,
                                                            expand_bifpn=expand_bifpn,
                                                            repeat=repeat_bifpn,
                                                            bifpn_channels=bifpn_channels,
                                                            factor2=factor2)

        self.scores = nn.Conv2d(self.num_classes, 5, 1, groups=5)

    def forward(self, x):
        _, _, height, width = x.shape
        feats = self.backbone(x)

        scores = self.scores(feats)
        scores = torch.sigmoid(scores)

        score_map = scores[:, :1]
        geo_height = scores[:, 1:3] * height  # top and bottom
        geo_width = scores[:, 3:] * width  # left and right
        return torch.cat((score_map, geo_height, geo_width), dim=1)


class MultiLabelEAST(nn.Module):
    def __init__(self,
                 num_labels: int = 2,
                 advprop: bool = True,
                 compound_coef: int = 4,
                 expand_bifpn: bool = False,
                 n_seg_channels: int = 40,
                 repeat_bifpn: int = 3,
                 bifpn_channels: int = 128,
                 geo_layers: int = 1,
                 class_layers: int = 1,
                 factor2: bool = False):
        super().__init__()
        self.num_classes = n_seg_channels
        self.backbone = EfficientDetForSemanticSegmentation(advprop=advprop,
                                                            num_classes=self.num_classes,
                                                            apply_sigmoid=False,
                                                            compound_coef=compound_coef,
                                                            expand_bifpn=expand_bifpn,
                                                            repeat=repeat_bifpn,
                                                            bifpn_channels=bifpn_channels,
                                                            factor2=factor2)
        self.num_labels = num_labels
        self.geo_scores = SegmentationClasssificationHead(self.num_classes, 4, num_layers=geo_layers, apply_sigmoid=True)
        #nn.Conv2d(self.num_classes, 4, 1, groups=4)
        self.class_scores = SegmentationClasssificationHead(self.num_classes, self.num_labels, num_layers=class_layers)
        # nn.Conv2d(
        #     self.num_classes, self.num_labels, kernel_size=1)

    def forward(self, x):
        _, _, height, width = x.shape
        feats = self.backbone(x)

        geo_scores = self.geo_scores(feats)
        class_scores = self.class_scores(feats)

        geo_height = geo_scores[:, :2] * height  # top and bottom
        geo_width = geo_scores[:, 2:] * width  # left and right
        return torch.cat((class_scores, geo_height, geo_width), dim=1)
