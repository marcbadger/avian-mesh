import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

import _init_paths
import torch
from torchvision import transforms as T
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNN

from pycocotools.coco import COCO
from utils.img_utils import create_color_mask


parser = argparse.ArgumentParser()
parser.add_argument('--root', default='data/cowbird/images', help='Path to image folder')
parser.add_argument('--annfile', default='data/cowbird/annotations/instance_train.json', help='Path to annotation')
parser.add_argument('--index', type=int, default=8, help='Index to the dataset for an example')
parser.add_argument('--outdir', type=str, default='examples_detection', help='Folder for output images')

if __name__ == '__main__':
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = args.root
    annfile = args.annfile

    # Load a maskRCNN finetuned on our birds
    network_transform = GeneralizedRCNNTransform(800, 1333, (0,0,0), (1,1,1))
    backbone = resnet_fpn_backbone(backbone_name='resnet101', pretrained=False)
    model = MaskRCNN(backbone, num_classes=2)
    model.transform = network_transform
    model.eval()
    model.load_state_dict(torch.load('models/detector.pth'))
    model.to(device)

    # Load a data split
    normalize = T.Normalize(
        mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.]
        )
    coco = COCO(annfile)

    # Load an image example
    available_Ids = coco.getImgIds()
    imgfile = coco.loadImgs(available_Ids[args.index])[0]['file_name']
    imgpath = root + '/' + imgfile

    img = cv2.imread(imgpath)
    img = normalize(torch.tensor(img).float().permute(2,0,1))

    # Run detector
    with torch.no_grad():
        output = model([img.to(device)])[0]

    # Visualization
    confident = (output['scores'] > 0.80)
    bird = (output['labels'] == 1)
    select = confident * bird
    masks = output['masks'][select].squeeze(1) > 0.75
    masks = masks.cpu()
    valid_mask = masks.sum(dim=0)>0

    color_masks = []
    for mask in masks:
        color_mask = create_color_mask(mask)
        color_masks.append(color_mask)
    color_masks = np.stack(color_masks, axis=0).sum(axis=0)

    img = cv2.imread(imgpath)[:,:,[2,1,0]]
    img[valid_mask] = 0.4*img[valid_mask] + 0.6*color_masks[valid_mask]

    # Save results
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    plt.imsave(args.outdir+'/{}.png'.format(args.index), img)


