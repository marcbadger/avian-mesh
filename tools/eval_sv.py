import argparse
import numpy as np
import torch
import torchvision.transforms as T

import _init_paths
from models import bird_model, load_regressor
from optimization import OptimizeSV, base_renderer
from keypoint_detection import load_detector, postprocess

from datasets import Cowbird_Dataset
from utils.geometry import perspective_projection
from utils.renderer import Silhouette_Renderer
from utils.evaluation import evaluate_pck, evaluate_iou
from utils.vis_bird import pose_bird

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='data/cowbird/images', help='Path to image folder')
parser.add_argument('--annfile', default='data/cowbird/annotations/instance_test.json', help='Path to annotation')
parser.add_argument('--use_mask', action='store_true', help='Whether masks are used in optimization')

def evaluate_singleview(root, annfile, device, use_mask=False):
    """
    Function to evaluation singleview reconstruction
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
        optimizer = OptimizeSV(num_iters=100, prior_weight=1, mask_weight=1, 
                               use_mask=False, device=device)

    # Dataset to run on
    normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])

    dataset = Cowbird_Dataset(root=root, annfile=annfile, scale_factor=0.25, transform=normalize)
    loader = torch.utils.data.DataLoader(dataset, batch_size=30)
    Pose_, Tran_, Bone_ = [], [], []
    GT_kpts, GT_masks, Sizes = [], [], []

    # Run reconstruction
    for i, (imgs, gt_kpts, gt_masks, meta) in enumerate(loader):
        print('Running on batch:', i+1)
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
        Pose_.append(pose_op)
        Tran_.append(tran_op)
        Bone_.append(bone_op)
        GT_kpts.append(gt_kpts)
        GT_masks.append(gt_masks)
        Sizes.append(meta['size'])

    Pose_ = torch.cat(Pose_)
    Tran_ = torch.cat(Tran_)
    Bone_ = torch.cat(Bone_)
    GT_kpts = torch.cat(GT_kpts)
    GT_masks = torch.cat(GT_masks)
    Sizes = torch.cat(Sizes)

    # Render reprojected kpts and masks
    kpts_3d, vertices = pose_bird(bird, Pose_[:,:3], Pose_[:,3:], Bone_, Tran_, pose2rot=True)
    kpts_2d = perspective_projection(kpts_3d, None, None, focal_length=2167, camera_center=128)
    faces = torch.tensor(bird.dd['F'])

    masks = []
    mask_renderer = Silhouette_Renderer(focal_length=2167, center=(128,128), img_w=256, img_h=256)
    for i in range(len(vertices)):
        m = mask_renderer(vertices[i], faces)
        masks.append(m)
    masks = torch.tensor(np.stack(masks)).long()


    # Evaluation
    PCK05, PCK10 = evaluate_pck(kpts_2d[:,:12,:], GT_kpts, size=Sizes)
    IOU = evaluate_iou(masks, GT_masks)

    avg_PCK05 = torch.mean(torch.stack(PCK05))
    avg_PCK10 = torch.mean(torch.stack(PCK10))
    avg_IOU = torch.mean(torch.stack(IOU))

    print('Average PCK05:', avg_PCK05)
    print('Average PCK10:', avg_PCK10)
    print('Average IOU:', avg_IOU)


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    evaluate_singleview(args.root, args.annfile, device, args.use_mask)

