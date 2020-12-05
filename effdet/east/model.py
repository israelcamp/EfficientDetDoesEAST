# Author: israel & dscarmo
# Modified From: Zylo11

import torch
from torch import nn

from ..segmentation import EfficientDetForSemanticSegmentation

class EfficientDetDoesEAST(nn.Module):

    def __init__(self, advprop=True, compound_coef=4, expand_bifpn=False, factor2=False):
        super().__init__()
        self.num_classes = 40
        self.backbone = EfficientDetForSemanticSegmentation(advprop=advprop,
                                                            num_classes=self.num_classes, 
                                                            apply_sigmoid=False,
                                                            compound_coef=compound_coef,
                                                            expand_bifpn=expand_bifpn,
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
