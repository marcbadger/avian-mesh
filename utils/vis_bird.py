import torch
import torchvision.transforms as T
import numpy as np
import cv2

from .renderer import Renderer


def pose_bird(bird, orient, pose, bone, tran, scale=1, pose2rot=True):
    
    bird_output = bird(orient, pose, bone, scale=scale, pose2rot=pose2rot)
    
    # from tran~[0,1] to tran_xyz of real unit
    y = -1
    z = 180
    tran_xyz = tran.clone()
    tran_xyz[:,1] = tran_xyz[:,1] + y
    tran_xyz[:,2] = tran_xyz[:,2]*18 + z
    tran_xyz = tran_xyz[:, None, :]

    keypoints_3d = bird_output['keypoints'] + tran_xyz
    vertices = bird_output['vertices'] + tran_xyz
    
    return keypoints_3d, vertices


def render_sample(bird, vertices, focal=2167, size=256, center=None, background=None):
    if background is None:
        background = np.ones([size, size, 3], dtype=np.uint8) * 255
        
    if isinstance(background, torch.Tensor):
        background = background.numpy().astype(np.uint8)
        
    if center is None:
        center = (size/2, size/2)

    renderer = Renderer(focal, center, img_w=size, img_h=size, faces=bird.dd['F'])
    img, depth = renderer(vertices, np.eye(3), [0,0,0], background)
    mask = depth > 0
    
    return img, mask


    