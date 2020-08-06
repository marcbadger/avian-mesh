import os
import torch
import torch.nn.functional as F

from .default import _C as cfg
from .pose_hrnet import get_pose_net



def load_detector(device='cpu'):
    """ utility function to load a trained detector
    """
    this_dir = os.path.dirname(__file__)
    cfg.merge_from_file(this_dir + '/w32_256x256.yaml')

    # pretrain state_dict by us
    state_dict = torch.load(this_dir + '/model_best.pth', map_location=device)

    model = get_pose_net(cfg, is_train=False)
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def postprocess(output, device=None):
    """ utility function to process output from HRNet: kpts+mask
        accept batch input/output
    """

    batch_size = output.shape[0]
    num_kpts = output.shape[1] - 2
    h, w = output.shape[2:]
    if device is None:
        device = output.device

    #### mask
    pred_mask = F.softmax(output[:,-2:,:,:], dim=1)[:,[1],:,:]
    pred_mask = F.interpolate(pred_mask, size=256).to(device)

    #### keypoints
    pred_kpts = output[:, :-2, :, :]
    pred_kpts = pred_kpts.reshape(batch_size, num_kpts, -1)

    val, ind = torch.max(pred_kpts, dim=2)
    pred_kpts = torch.zeros([batch_size, num_kpts, 3])
    pred_kpts[:,:,0] = 4*(ind%w)
    pred_kpts[:,:,1] = 4*(ind//w)
    pred_kpts[:,:,2] = val
    pred_kpts = pred_kpts.to(device)

    return pred_kpts, pred_mask


