import yaml
import os
import numpy as np
import torch

from .renderer import Renderer
from .geometry import perspective_projection


def get_fullsize_masks(masks, bboxes, h=1200, w=1920):
    full_masks = []
    for i in range(len(masks)):
        box = bboxes[i]
        full_mask = torch.zeros([h, w], dtype=torch.bool)
        full_mask[box[1]:box[1]+box[3]+1, box[0]:box[0]+box[2]+1] = masks[i]
        full_masks.append(full_mask)
    full_masks = torch.stack(full_masks)

    return full_masks


def get_cam(frames, device='cpu'):
    '''
    Input:
    frames: list containing frame numbers
    
    '''
    # map frames to cams
    frame_to_cam_map = {0:3,1:2,2:7,3:6,4:0,5:1,6:4,7:5}
    cams = [frame_to_cam_map[i] for i in frames]  
    
    # cam parameters from yaml
    this_dir = os.path.dirname(__file__)

    calibs = yaml.safe_load(open(this_dir + '/extrinsic_calib_v2.yaml'))
    cam_Ps = [np.array(calibs[key]['T_cam_imu']) for key in sorted(calibs)]
    cam_in = [np.array(calibs[key]['intrinsics']) for key in sorted(calibs)]
    cam_dt = [np.array(calibs[key]['distortion_coeffs']) for key in sorted(calibs)]
    
    # rotation, translation, focal_length, camera_center
    rotation = [cam_Ps[i][:3, :3] for i in cams]
    translation = [cam_Ps[i][:3, 3] for i in cams]
    focal = [cam_in[i][:2].mean() for i in cams]
    center = [cam_in[i][2:] for i in cams]
    distortion = [cam_dt[i] for i in cams]
    
    # Convert to tensor
    rotation = torch.tensor(rotation).float().to(device)
    translation = torch.tensor(translation).float().to(device)
    focal = torch.tensor(focal).float().to(device)
    center = torch.tensor(center).float().to(device)
    distortion = torch.tensor(distortion).float().to(device)
    
    return rotation, translation, focal, center, distortion


def projection_loss(x, y):
    loss = (x.float() - y.float()).norm(p=2)
    return loss


def triangulation_LBFGS(x, R, t, focal, center, distortion, device='cpu'):
    n = x.shape[0]
    X = torch.tensor([2.5, 1.2, 1.95])[None,None,:]
    X.requires_grad_()
    
    x = x.to(device)
    X = X.to(device)
    
    losses = []
    optimizer = torch.optim.LBFGS([X], lr=1, max_iter=100, line_search_fn='strong_wolfe')
    
    def closure():
        projected_points = perspective_projection(X.repeat(n,1,1), R, t, focal, center, distortion)
        loss = projection_loss(projected_points.squeeze(), x)
        
        optimizer.zero_grad()
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    with torch.no_grad():
        projected_points = perspective_projection(X.repeat(n,1,1), R, t, focal, center, distortion)
        loss = projection_loss(projected_points.squeeze(), x)
        losses.append(loss.detach().item())    
    X = X.detach().squeeze()
    
    return X, losses


def triangulation(x, R, t, focal, center, distortion, device='cpu'):
    n = x.shape[0]
    X = torch.tensor([2.5, 1.2, 1.95])[None,None,:]
    X.requires_grad_()
    
    x = x.to(device)
    X = X.to(device)
    
    losses = []
    optimizer = torch.optim.Adam([X], lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 90], gamma=0.1)
    for i in range(100):
        projected_points = perspective_projection(X.repeat(n,1,1), R, t, focal, center, distortion)
        loss = projection_loss(projected_points.squeeze(), x)
    
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.detach().item())
        
    X = X.detach().squeeze()
    
    return X, losses


def get_gt_3d(keypoints, frames, LBFGS=False):
    '''
    Input: 
        keypoints (bn, kn, 2): 2D kpts from different views
        frames (bn): frame numbers
    Output:
        kpts_3d (kn, 4): ground truth 3D kpts, with validility

    '''
    bn, kn, _ = keypoints.shape
    kpts_3d = torch.zeros([kn, 4])
    
    # 
    rotation, translation, focal, center, distortion = get_cam(frames)
    kpts_valid = []
    cams = []
    for i in range(kn):
        valid = keypoints[:, i, -1]>0
        kpts_valid.append(keypoints[valid, i, :2])
        cams.append(valid)
    
    #
    for i in range(kn):
        x = kpts_valid[i]
        if len(x)>=2:
            R = rotation[cams[i]]
            t = translation[cams[i]]
            f = focal[cams[i]]
            c = center[cams[i]]
            dis = distortion[cams[i]]
            
            if LBFGS:
                X, _ = triangulation_LBFGS(x, R, t, f, c, dis)
            else:
                X, _ = triangulation(x, R, t, f, c, dis)
                
            kpts_3d[i,:3] = X
            kpts_3d[i,-1] = 1
            
    return kpts_3d


def Procrustes(X, Y):
    """ 
    Solve full Procrustes: Y = s*RX + t

    Input:
        X (N,3): tensor of N points
        Y (N,3): tensor of N points in world coordinate 
    Returns:
        R (3x3): tensor describing camera orientation in the world (R_wc)
        t (3,): tensor describing camera translation in the world (t_wc)
        s (1): scale
    """
    # remove translation
    A = Y - Y.mean(dim=0, keepdim=True)
    B = X - X.mean(dim=0, keepdim=True)
    
    # remove scale
    sA = (A*A).sum()/A.shape[0]
    sA = sA.sqrt()
    sB = (B*B).sum()/B.shape[0]
    sB = sB.sqrt()   
    A = A / sA
    B = B / sB
    s = sA / sB
    
    # to numpy, then solve for R
    A = A.t().numpy()
    B = B.t().numpy()
    
    M = B @ A.T
    U, S, VT = np.linalg.svd(M)
    V = VT.T
    
    d = np.eye(3)
    d[-1, -1] = np.linalg.det(V @ U.T)
    R = V @ d @ U.T
    
    # back to tensor
    R = torch.tensor(R).float()
    t = Y.mean(axis=0) - R@X.mean(axis=0) * s
    
    return R, t, s
    
    
def get_renderer_list(faces):
    all_frames = [0, 1, 2, 3, 4, 5, 6, 7]
    cam_rot, cam_t, focal, center, distortion = get_cam(all_frames)
    renderer_list = []
    for i in range(len(all_frames)):
        renderer = Renderer(focal[i], center[i], img_w=1920, img_h=1200, faces=faces)
        renderer_list.append(renderer)
        
    return renderer_list


def render_vertex_on_frame(img, vertex_posed, bird, frames):

    rotation, translation, focal, center, distortion = get_cam(frames)

    # Extrinsic
    points = torch.einsum('bij,bkj->bki', rotation, vertex_posed) + translation

    # Distortion
    if distortion is not None:
        kc = distortion
        d = points[:,:,2:]
        points = points[:,:,:] / points[:,:,2:]

        r2 = points[:,:,0]**2 + points[:,:,1]**2
        dx = (2 * kc[:,[2]] * points[:,:,0] * points[:,:,1] 
              + kc[:,[3]] * (r2 + 2*points[:,:,0]**2))

        dy = (2 * kc[:,[3]] * points[:,:,0] * points[:,:,1] 
              + kc[:,[2]] * (r2 + 2*points[:,:,1]**2))

        x = (1 + kc[:,[0]]*r2 + kc[:,[1]]*r2.pow(2) + kc[:,[4]]*r2.pow(3)) * points[:,:,0] + dx
        y = (1 + kc[:,[0]]*r2 + kc[:,[1]]*r2.pow(2) + kc[:,[4]]*r2.pow(3)) * points[:,:,1] + dy

        points = torch.stack([x, y, torch.ones_like(x)], dim=-1) * d

    points = points[0]

    # Rendering
    renderer = Renderer(focal, center[0], img_w=1920, img_h=1200, faces=bird.dd['F'])
    img_pose, _ = renderer(points, np.eye(3), [0,0,0], img)
    img_pose = img_pose.astype(np.uint8)

    return img_pose


def render_mesh(bird, pose_est, bone_est, scale_est=1, camera_t= torch.tensor([[2, -7, 35]]).float()):
    # Background
    background = torch.ones([1000, 1500, 3]).float()

    # Camera parameters
    #camera_t = torch.tensor([[2, -7, 35]]).float()
    camera_center = torch.tensor([[1500//2, 1000//2]]).float()
    focal_length = 2165

    # Bird Mesh
    bird_output = bird(pose_est[:, 0:3], pose_est[:, 3:], bone_est, scale_est)
    vertex_posed = bird_output['vertices']
    #vertex_posed += torch.tensor([[[0,10,8]]]).float()

    # Rendering
    renderer = Renderer(focal_length=focal_length, center=(750, 500), img_w=1500, img_h=1000, faces=bird.dd['F'])
    img_1, _ = renderer(vertex_posed[0].clone().numpy(), np.eye(3), camera_t[0].clone().numpy(), background.clone().numpy())


    # Render: Second View
    aroundy = cv2.Rodrigues(np.array([0, np.radians(45.), 0]))[0]
    center = vertex_posed.numpy()[0].mean(axis=0)
    rot_vertices = np.dot((vertex_posed.numpy()[0] - center), aroundy) + center
    img_2, _ = renderer(rot_vertices, np.eye(3), camera_t[0].clone().numpy(), background.clone().numpy())

    # Render: Third View
    aroundy = cv2.Rodrigues(np.array([0, np.radians(-45.), 0]))[0]
    center = vertex_posed.numpy()[0].mean(axis=0)
    rot_vertices = np.dot((vertex_posed.numpy()[0] - center), aroundy) + center
    img_3, _ = renderer(rot_vertices, np.eye(3), camera_t[0].clone().numpy(), background.clone().numpy())

    return [img_1, img_2, img_3]

