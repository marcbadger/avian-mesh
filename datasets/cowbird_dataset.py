import os
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.mask import decode

from utils.img_utils import dialate_boxes

class Cowbird_Dataset(torch.utils.data.Dataset):
    """
    Dataset class for instance level task, including detection, instance segmentation, 
    and single view reconstruction. Since data are in COCO format, this class utilize
    COCO API to do most of the dataloading.
    """
    def __init__(self, root, annfile, scale_factor=0.25, output_size=256, transform=None):
        self.root = root
        self.coco = COCO(annfile)
        self.imgIds = self.coco.getImgIds(catIds=1)
        self.imgIds.sort()
        
        self.scale_factor = scale_factor
        self.output_size = output_size
        self.transform = transform
        self.data = self.get_data()
        
    def __getitem__(self, index):
        data = self.data[index]
        x, y, w, h = data['bbox']

        # input image
        img = cv2.imread(data['imgpath'])
        img = img[y:y+h, x:x+w]
        img = cv2.resize(img, (self.output_size, self.output_size))
        
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.tensor(img).permute(2,0,1).float()/255

        # keypoints
        kpts = data['keypoints'].clone()
        valid = kpts[:,-1] > 0
        kpts[valid,:2] -= torch.tensor([x, y])
        kpts[valid,:2] *= self.output_size / w.float()
        
        # mask
        mask = decode(data['rle'])
        mask = mask[y:y+h, x:x+w]
        mask = cv2.resize(mask, (self.output_size, self.output_size))
        mask = torch.tensor(mask).long()
        
        # meta
        size = data['size'] * self.output_size / w.float()
        meta = {
            'imgpath': data['imgpath'],
            'size': size
            }
                
        return img, kpts, mask, meta
    
    def __len__(self):
        return len(self.data)
    
    def get_data(self):
        data = []
        for imgId in self.imgIds:
            data.extend(self.load_data(imgId))
        return data
    
    def load_data(self, imgId):
        img_dict = self.coco.loadImgs(imgId)[0]
        width = img_dict['width']
        height = img_dict['height']
        
        annIds = self.coco.getAnnIds(imgIds=imgId)
        anns = self.coco.loadAnns(annIds)
        data = []
        for ann in anns:
            path = self.path_from_Id(imgId)
            kpts = torch.tensor(ann['keypoints']).float().reshape(-1, 3)
            bbox = dialate_boxes([ann['bbox']], s=self.scale_factor)[0]
            rle  = self.coco.annToRLE(ann)
            size = max(ann['bbox'][2:])
            
            data.append({
                'imgpath': path,
                'bbox': bbox,
                'keypoints': kpts,
                'rle': rle,  # to save memory, we store rle and convert to mask on the fly 
                'size': size
            })
            
        return data
            
    def path_from_Id(self, imgId):
        img_dict = self.coco.loadImgs(imgId)[0]
        filename = img_dict['file_name']
        path = os.path.join(self.root, filename)
        return path

