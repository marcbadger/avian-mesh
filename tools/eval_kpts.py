import argparse
import torch
import torchvision.transforms as T

import _init_paths
from datasets import Cowbird_Dataset
from keypoint_detection import load_detector, postprocess
from utils.evaluation import evaluate_pck, evaluate_iou

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='data/cowbird/images', help='Path to image folder')
parser.add_argument('--annfile', default='data/cowbird/annotations/instance_test.json', help='Path to annotation')

def evaluate_detector(root, annfile, device):
    """
    Function to evaluation keypoint and mask prediction
    """

    normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])

    dataset = Cowbird_Dataset(root=root, annfile=annfile, scale_factor=0.25, transform=normalize)
    loader = torch.utils.data.DataLoader(dataset, batch_size=30)
    detector = load_detector(device=device)

    PCK05 = []
    PCK10 = []
    IOU = []
    for i, (imgs, kpts, masks, meta) in enumerate(loader):
        print('Evaluating batch:', i+1)
        with torch.no_grad():
            # Prediction
            output = detector(imgs.to(device))
            pred_kpts, pred_masks = postprocess(output, 'cpu')
            pred_masks_long = (pred_masks.squeeze(1)>0.5).long()
            
        pck05, pck10 = evaluate_pck(pred_kpts, kpts, size=meta['size'])
        iou = evaluate_iou(pred_masks_long, masks)
        PCK05.extend(pck05)
        PCK10.extend(pck10)
        IOU.extend(iou)

    avg_PCK05 = torch.mean(torch.stack(PCK05))
    avg_PCK10 = torch.mean(torch.stack(PCK10))
    avg_IOU = torch.mean(torch.stack(IOU))
    print('Average PCK05:', avg_PCK05)
    print('Average PCK10:', avg_PCK10)
    print('Average IOU:', avg_IOU)


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    evaluate_detector(args.root, args.annfile, device)

