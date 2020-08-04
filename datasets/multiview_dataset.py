import os
import torch

class Multiview_Dataset(torch.utils.data.Dataset):
    """
    Dataset class to read bird instances that have multiview matches.
    Each index outputs one instance and its multiview annotations
    """
    def __init__(self, root='data/cowbird/images', 
                 annfile='data/cowbird/annotations/multiview_instance.pth'):
        self.root = root
        self.anns = torch.load(annfile)
        
    def __getitem__(self, index):
        ann = self.anns[index]
        masks = self.get_fullsize_masks(ann['masks'], ann['bboxes'])
            
        data = {
            'img_ids': ann['img_ids'],
            'imgpaths': [os.path.join(self.root, file) for file in ann['img_filenames']],
            'frames': ann['frames'],
            'bboxes': ann['bboxes'],
            'keypoints': ann['keypoints'].float(),
            'masks': masks
        }
        
        return data

    def __len__(self):
        return len(self.anns)
    
    def get_fullsize_masks(self, masks, bboxes, h=1200, w=1920):
        full_masks = []
        for i in range(len(masks)):
            box = bboxes[i]
            full_mask = torch.zeros([h, w], dtype=torch.bool)
            full_mask[box[1]:box[1]+box[3]+1, box[0]:box[0]+box[2]+1] = masks[i]
            full_masks.append(full_mask)
        full_masks = torch.stack(full_masks)

        return full_masks
        