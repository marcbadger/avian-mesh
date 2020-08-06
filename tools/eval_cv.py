import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms as T

import _init_paths
from models import bird_model, load_regressor
from optimization import OptimizeSV, base_renderer
from keypoint_detection import load_detector, postprocess

from datasets import Multiview_Dataset
from utils import multiview
from utils import multiview_utils as mutils
from utils.evaluation import evaluate_pck, evaluate_iou
from utils.img_utils import dialate_boxes


parser = argparse.ArgumentParser()
parser.add_argument('--use_mask', action='store_true', help='Whether masks are used in optimization')

def evaluate_crossview(device, use_mask=False):
    """
    Function to evaluation single reconstruction through crossview validation
    """

    # Models and optimizer
    bird = bird_model()
    predictor = load_detector().to(device)
    regressor = load_regressor().to(device)

    if args.use_mask:
        if device == 'cpu':
            print('Warning: using mask during optimization without GPU acceleration is very slow!')
        silhouette_renderer = base_renderer(size=256, focal=2167, device=device)
        optimizer = OptimizeSV(num_iters=100, prior_weight=1, mask_weight=1, 
                               use_mask=True, renderer=silhouette_renderer, device=device)
        print('Using mask for single view optimization')
    else:
        optimizer = OptimizeSV(num_iters=100, prior_weight=10, mask_weight=1, 
                               use_mask=False, device=device)

    # Dataset to run on
    normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])

    multiview_data = Multiview_Dataset()
    renderer_list = mutils.get_renderer_list(bird.dd['F'])
    IOU = []
    PCK05 = []
    PCK10 = []

    # Run singleview reconstruction
    for i, sample in enumerate(multiview_data):
        print('Running on sample:', i+1)
        frames = sample["frames"]
        img_filenames = sample["imgpaths"]
        keypoints = sample["keypoints"]
        masks = sample['masks']
        bboxes = sample["bboxes"]
        dialated_bboxes = dialate_boxes(bboxes)

        imgs = []
        masks_gt = []
        kpts_gt = []
        for j in range(len(frames)):
            box = dialated_bboxes[j]
            img = cv2.imread(img_filenames[j])
            img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
            img = cv2.resize(img, dsize=(256,256))
            img = normalize(img)
            imgs.append(img)
        imgs = torch.stack(imgs)


        with torch.no_grad():
            # Prediction
            output = predictor(imgs.to(device))
            pred_kpts, pred_mask = postprocess(output)

            # Regression
            kpts_in = pred_kpts.reshape(pred_kpts.shape[0], -1)
            mask_in = pred_mask
            p_est, b_est = regressor(kpts_in, mask_in)
            pose, tran, bone = regressor.postprocess(p_est, b_est)

        # Optimization
        ignored = pred_kpts[:, :, 2] < 0.3
        opt_kpts = pred_kpts.clone()
        opt_kpts[ignored] = 0
        pose_op, bone_op, tran_op, model_mesh = optimizer(pose, bone, tran, 
                                              focal_length=2167, camera_center=128, 
                                              keypoints=opt_kpts, masks=mask_in.squeeze(1))


        # Global rigid alignment with reconstruction from each view
        for j in range(len(frames)):
            vertex_posed, mesh_keypoints \
            = multiview.multiview_rigid_alignment(bird, pose_op[[j], 3:], bone_op[[j]], 
                                                  keypoints, frames, num_iters=100, device='cpu')
            proj_masks = multiview.reproject_masks(vertex_posed, renderer_list, frames)
            proj_kpts = multiview.reproject_keypoints(mesh_keypoints, frames)

            iou = evaluate_iou(proj_masks, masks)
            pck05, pck10 = evaluate_pck(proj_kpts, keypoints, bboxes)

            # Removed jth sample to accord with the published metrics:
            # "average across all non-source views"
            iou.pop(j)
            pck05.pop(j)
            pck10.pop(j)
            
            iou = torch.stack(iou).mean()
            pck05 = torch.stack(pck05).mean()
            pck10 = torch.stack(pck10).mean()
            IOU.append(iou)
            PCK05.append(pck05)
            PCK10.append(pck10)



    avg_PCK05 = torch.mean(torch.stack(PCK05))
    avg_PCK10 = torch.mean(torch.stack(PCK10))
    avg_IOU = torch.mean(torch.stack(IOU))

    print('Average PCK05:', avg_PCK05)
    print('Average PCK10:', avg_PCK10)
    print('Average IOU:', avg_IOU)


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    evaluate_crossview(device, args.use_mask)

