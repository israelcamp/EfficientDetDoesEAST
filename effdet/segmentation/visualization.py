import torchvision as tv
import imgaug.augmentables as ia
import torch
import matplotlib.pyplot as plt

def visualize_batch(img, mask):
    grid_image = tv.utils.make_grid(img, nrow=8)
    iam = ia.SegmentationMapsOnImage(mask.permute(1,2,0).numpy(), shape=mask.shape[1:])
    iam = iam.resize(768) # TODO: this needs to be an argument
    mask = torch.tensor(iam.arr).permute(2, 0, 1)

    grid_mask = tv.utils.make_grid(mask.unsqueeze(1), nrow=8)

    plt.figure(figsize=(20,20))
    grid_image = (grid_image + 1)/2
    plt.imshow(grid_image.permute(1,2,0))
    plt.imshow(grid_mask.permute(1,2,0)*255, alpha=.5)
