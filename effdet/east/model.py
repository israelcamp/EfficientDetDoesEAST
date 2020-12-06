# Author: israel & dscarmo
# Modified From: Zylo11

import torch
from torch import nn

from ..segmentation import EfficientDetForSemanticSegmentation

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
