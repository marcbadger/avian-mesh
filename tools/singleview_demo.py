import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as T

import _init_paths
from models import bird_model, load_regressor
from optimization import OptimizeSV, base_renderer

from datasets import Cowbird_Dataset
from keypoint_detection import load_detector, postprocess
from utils.vis_bird import render_sample


parser = argparse.ArgumentParser()
parser.add_argument('--root', default='data/cowbird/images', help='Path to image folder')
parser.add_argument('--annfile', default='data/cowbird/annotations/instance_test.json', help='Path to annotation')
parser.add_argument('--index', type=int,  default=0, help='Index in the dataset for example reconstruction')
parser.add_argument('--use_mask', action='store_true', help='Whether masks are used in optimization')
parser.add_argument('--outdir', type=str, default='examples', help='Folder for output images')

if __name__ == '__main__':
    args = parser.parse_args()

    # Load model and optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bird = bird_model()
    predictor = load_detector().to(device)
    regressor = load_regressor().to(device)
    print('Device used:', device)

    if args.use_mask:
        if device == 'cpu':
            print('Warning: using mask during optimization without GPU acceleration is very slow!')
        silhouette_renderer = base_renderer(size=256, focal=2167, device=device)
        optimizer = OptimizeSV(num_iters=100, prior_weight=1, mask_weight=1, 
                               use_mask=True, renderer=silhouette_renderer, device=device)
        print('Using mask for optimization')
    else:
        optimizer = OptimizeSV(num_iters=100, prior_weight=1, mask_weight=1, 
                               use_mask=False, device=device)

    # Load dataset
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])
    unnormalize = T.Compose([
        T.Normalize(mean=[0, 0, 0], std=[1/0.225, 1/0.224, 1/0.229]),
        T.Normalize(mean=[-0.406, -0.456, -0.485], std=[1, 1, 1])
        ])
    valid_set = Cowbird_Dataset(args.root, args.annfile, transform=normalize)


    # Run on the indexed sample from the validset
    imgs, target_kpts, target_masks, meta = valid_set[args.index]
    imgs = imgs[None]

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


    # Save reconstruction results
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    for i, img in enumerate(imgs):
        img_out = unnormalize(img)
        img_out = img_out.permute(1,2,0)[:,:,[2,1,0]] * 255
        img_opt, _ = render_sample(bird, model_mesh[i], background=img_out)

        img_save = np.zeros([256, 256*2, 3]).astype(np.uint8)
        img_save[:, 256*0:256*(0+1), :] = img_out
        img_save[:, 256*1:256*(1+1), :] = img_opt

        plt.imsave(args.outdir+'/{:02d}.png'.format(i), img_save) 






