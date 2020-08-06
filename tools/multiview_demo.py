import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

import _init_paths
from models import bird_model
from optimization import OptimizeMV

from datasets import Multiview_Dataset
from utils import multiview
from utils import multiview_utils as mutil


parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, default=44, help='Index for an example in the multiview dataset')
parser.add_argument('--outdir', type=str, default='examples_multiview', help='Folder for output images')

if __name__ == '__main__':
    args = parser.parse_args()

    # Load model and optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    bird = bird_model(mesh='bird_eccv.json')
    optimizer = OptimizeMV(num_iters=300, lim_weight=100, prior_weight=100, 
                           bone_weight=200, device=device, mesh='bird_eccv.json')

    multiview_data = Multiview_Dataset()
    sample = multiview_data[args.index]
    frames = sample["frames"]
    img_filenames = sample["imgpaths"]
    keypoints = sample["keypoints"]

    vertex_posed, mesh_keypoints, t, body_pose, bone, scale \
    = multiview.fit_mesh(bird, optimizer, keypoints, frames, device=device)


    # Save reconstruction results
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    for i in range(len(img_filenames)):
        frame = [frames[i]]
        img = plt.imread(img_filenames[i])
        img_pose = mutil.render_vertex_on_frame(img, vertex_posed, bird, frame)

        plt.imsave(args.outdir+'/view_{}.png'.format(i), img_pose) 






