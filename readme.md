# EfficientDet For Semantic Segmentation and EAST

## Where did I copy it from?

1. [Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) by Zylo117 contains most of the base code to create the EfficientDet, however I cleaned it a little to have only what I need and used [lukemela's implementation of EfficientNet directly](https://github.com/lukemelas/EfficientNet-PyTorch)
1. [EfficientDet](https://arxiv.org/abs/1911.09070) original paper
1. [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155) inspiration to this repo

## What can I do?

1. **Object detection** - you can still use EfficientDet for object detection, however we recommend sticking to Zylo's code
1. **Segmentation** - you can train EfficientDet to perform segmentation
1. **Text Detection** - you can use EfficientDet to segment and predict bounding boxes as proposed on [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155)

## Limitations

1. We do not have support for EfficientDet-D7x

## Results

1. Our first public result is on [ICDAR 2019 Robust Reading Challenge on Scanned Receipts OCR and Information Extraction](https://rrc.cvc.uab.es/?ch=13)
   - We obtained an _Hmean_ of 93.76% on test set
   - Training involved only a EfficientNet backbone initialized on ImageNet that was frozen. We trained the rest model from a random start.

1. The checkpoint for segmentation on [ICDAR2019 Robust Reading Challenge on Arbitrary-Shaped Text](https://rrc.cvc.uab.es/?ch=14) can be found [here](https://drive.google.com/file/d/1DI8BpWiCFti_qyu0Dw2ebPUSM6ebSkCl/view?usp=sharing). The experiment containing training code and hyper parameters can be found [here](https://ui.neptune.ai/israelcamp/TextDetection/e/TD-18/charts)
## TODO

- [x] Add other coeficients for **Segmentation** and **EAST**
- [x] Add D7 to **Segmentation** and **EAST**
- [ ] Add EfficientDet-D7x
- [ ] Release model checkpoint for the SROIE 2019 submission
- [x] Add results and model checkpoint for the [ICDAR2019 Robust Reading Challenge on Arbitrary-Shaped Text](https://rrc.cvc.uab.es/?ch=14)
- [ ] Add documentation
- [ ] Add testing
