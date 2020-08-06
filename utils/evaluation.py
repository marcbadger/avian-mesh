import torch

def evaluate_iou(proj_masks, masks):
    IOU = []
    for proj_mask, mask in zip(proj_masks, masks):
        
        stack = torch.stack([mask, proj_mask]).byte()
        I = torch.all(stack, 0).sum([0,1]).float()
        U = torch.any(stack, 0).sum([0,1]).float()
        IOU.append(I/U)
        
    return IOU


def evaluate_pck(proj_kpts, keypoints, bboxes=None, size=256):
    PCK05 = []
    PCK10 = []
        
    err = proj_kpts[:,:,:2] - keypoints[:,:,:2]
    err = err.norm(dim=2, keepdim=True)
    
    if bboxes is not None:
        maxHW, ind = torch.max(bboxes[:,2:], dim=1)
    else:
        if type(size) == int:
            maxHW = [size] * len(err)
        else:
            maxHW = size
        
    for i in range(len(err)):
        valid = keypoints[i, :, 2:] > 0
        err_i = err[i][valid]
        err_i = err_i / maxHW[i]
        pck05 = (err_i < 0.05).float().mean()
        pck10 = (err_i < 0.10).float().mean()
        PCK05.append(pck05)
        PCK10.append(pck10)
    
    return PCK05, PCK10

    