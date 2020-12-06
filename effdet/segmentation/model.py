
import torch
from torch import nn

from ..base.model import BiFPN, EfficientNet, SegmentationClasssificationHead
from ..base.enet_utils import MemoryEfficientSwish


class EfficientDetForSemanticSegmentation(nn.Module):

    def __init__(self, advprop=True, num_classes=2, apply_sigmoid=False, compound_coef=4, repeat=3, expand_bifpn=False, factor2=False, bifpn_channels=128):
        super().__init__()
        self.compound_coef = compound_coef
        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
        self.num_classes = num_classes
        self.expand_bifpn = expand_bifpn

        if factor2:
            conv_channel_coef = {
                # the channels of P2/P3/P4.
                0: [16, 24, 40],
                1: [16, 24, 40],
                2: [16, 24, 48],
                3: [24, 32, 48],
                4: [24, 32, 56],
                5: [24, 40, 64],
                6: [32, 40, 72],
                7: [32, 48, 80]
            }
        else:
            conv_channel_coef = {
                # the channels of P2/P3/P4.
                0: [24, 40, 112],
                1: [24, 40, 112],
                2: [24, 48, 112],
                3: [32, 48, 136],
                4: [32, 56, 160],
                5: [40, 64, 176],
                6: [40, 72, 200],
                7: [48, 80, 224]
            }

        if expand_bifpn:
            self.expand_conv = nn.Sequential(nn.ConvTranspose2d(bifpn_channels, bifpn_channels, 2, 2),
                                             nn.BatchNorm2d(bifpn_channels),
                                             MemoryEfficientSwish())

        self.bifpn = nn.Sequential(
            *[BiFPN(bifpn_channels,
                    conv_channel_coef[compound_coef],
                    True if i == 0 else False,
                    attention=True if self.compound_coef < 6 else False)
              for i in range(repeat)])

        self.classifier = SegmentationClasssificationHead(in_channels=bifpn_channels,
                                                          num_classes=self.num_classes,
                                                          num_layers=repeat,  # should it be repeat - 1?
                                                          apply_sigmoid=apply_sigmoid
                                                          )

        self.backbone_net = EfficientNet(self.backbone_compound_coef[self.compound_coef],
                                         advprop=advprop,
                                         factor2=factor2)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_backbone_bn(self):
        ''' Freezes only the BN of the backbone_net '''
        for m in self.backbone_net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def extract_backbone_features(self, inputs):
        max_size = inputs.shape[-1]

        p2, p3, p4, _ = self.backbone_net(inputs)

        features = (p2, p3, p4)
        return features

    def extract_bifpn_features(self, features):
        features = self.bifpn(features)
        return features

    def forward(self, inputs):
        features = self.extract_backbone_features(inputs)
        feat_map = self.extract_bifpn_features(features)[0]

        if self.expand_bifpn:
            feat_map = self.expand_conv(feat_map)

        classification = self.classifier(feat_map)

        return classification

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')
