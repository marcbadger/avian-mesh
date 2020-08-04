import numpy as np 
from torch.nn import functional as F
import torch

"""
Useful geometric operations, e.g. Perspective projection and a differentiable Rodrigues formula
Parts of the code are taken from https://github.com/MandyMo/pytorch_HMR and https://github.com/nkolot/SPIN
"""
def batch_rodrigues(theta, dtype=torch.float32):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """

    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat).float()

def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """ 
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat    

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

def perspective_projection(points, rotation, translation,
                           focal_length, camera_center, distortion=None):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
        distortion (bs, 5): distortion factors
    """
    batch_size = points.shape[0]
    
    # Extrinsic
    if rotation is not None:
        points = torch.einsum('bij,bkj->bki', rotation, points)

    if translation is not None:
        points = points + translation.unsqueeze(1)

    if distortion is not None:
        kc = distortion
        points = points[:,:,:2] / points[:,:,2:]
        
        r2 = points[:,:,0]**2 + points[:,:,1]**2
        dx = (2 * kc[:,[2]] * points[:,:,0] * points[:,:,1] 
                + kc[:,[3]] * (r2 + 2*points[:,:,0]**2))

        dy = (2 * kc[:,[3]] * points[:,:,0] * points[:,:,1] 
                + kc[:,[2]] * (r2 + 2*points[:,:,1]**2))
        
        x = (1 + kc[:,[0]]*r2 + kc[:,[1]]*r2.pow(2) + kc[:,[4]]*r2.pow(3)) * points[:,:,0] + dx
        y = (1 + kc[:,[0]]*r2 + kc[:,[1]]*r2.pow(2) + kc[:,[4]]*r2.pow(3)) * points[:,:,1] + dy
        
        points = torch.stack([x, y, torch.ones_like(x)], dim=-1)
        
        
    # Intrinsic
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Apply camera intrinsicsrf
    points = points / points[:,:,-1].unsqueeze(-1)
    projected_points = torch.einsum('bij,bkj->bki', K, points)
    projected_points = projected_points[:, :, :-1]

    return projected_points



