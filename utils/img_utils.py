import torch
from torch.nn import functional as F
import cv2
import numpy as np

from .geometry import perspective_projection
from .renderer import Renderer


def draw_kpts(img, kpts, r=5, thickness=5):
    if isinstance(img, np.ndarray):
        img = img.copy().astype(np.uint8)
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        img = img.copy().astype(np.uint8)
        
    for kpt in kpts:
        if len(kpt)>2:
            x, y, c = kpt
        else:
            x, y = kpt
            c = 1

        if c > 0:
            cv2.circle(img, (x, y), r, (255,0,0), thickness)

    return img


def dialate_boxes(bboxes, s=0.25, max_x=1920, max_y=1200, square=True, clip=True):
    '''
    Input:
    bboxes: Nx4 tensor
    
    '''
    if isinstance(bboxes, list):
        bboxes = torch.tensor(bboxes)
    bboxes = bboxes.type(torch.float)

    c = bboxes[:,:2] + bboxes[:,2:]/2
    ss = bboxes[:,2:]/2 * (1 + s)
    
    # take 1:1 box ratio if square
    if square:
        max_val, _ = ss.max(dim=1, keepdim=True)
        ss = ss/max_val
        ss.clamp_(1)
        ss = ss * max_val

    new_bboxes = torch.zeros(bboxes.shape, dtype=torch.int)
    new_bboxes[:,:2] = c - ss
    new_bboxes[:,2:] = c + ss

    if clip:
        new_bboxes[:,:2].clamp_(0)
        new_bboxes[:,2].clamp_(0, max_x)
        new_bboxes[:,3].clamp_(0, max_y)
    
    new_bboxes[:,2:] = new_bboxes[:,2:] - new_bboxes[:,:2]
    return new_bboxes


def expand_masks(masks, bboxes, h=1200, w=1920):
    '''
    Input:
    masks: list of 2D tensor, each could have different shape
    bboxes: Nx4 tensor
    
    '''
    new_masks = []
    for i in range(len(masks)):
        box = bboxes[i]
        mask = masks[i]
        
        mh, mw = mask.shape
        x = (box[2] - mw)/2
        y = (box[3] - mh)/2
        
        exp_mask = torch.zeros([box[3], box[2]], dtype=torch.bool)
        exp_mask[y:y+mh, x:x+mw] = mask
        new_masks.append(exp_mask)
    return new_masks


def resize_img(img, size=256):
    img = img.float()
    if img.dim() > 2 and img.shape[-1]==3:
        img = img.permute([2,0,1])
        img = img[None, :, :, :]
        img = F.interpolate(img, size)[0]
        img = img.permute([1,2,0]).byte()
    
    else:
        img = img[None, None, :, :]
        img = F.interpolate(img, size)[0, 0]
        img = img.byte()

    return img


def compare_mask(m1, m2):
    h, w = m1.shape
    valid_1 = (m1==True)
    valid_2 = (m2==True)
    img = torch.zeros([h,w,3])
    
    img[:,:,0][valid_1] = 0.8
    img[:,:,1][valid_2] = 0.8
    
    return img


def create_color_mask(mask):
    color = ((np.random.random((1, 3))*0.6+0.4)*255).astype(int).tolist()[0]
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3]).astype(int)
    color_mask[mask==1] = color
    return color_mask


