import cv2
import numpy as np
import torch

from optimization.losses import camera_fitting_loss
from utils.renderer import Renderer
from utils.geometry import perspective_projection
import utils.multiview_utils as mutils


def fit_geometry(bird, keypoints, frames, init_pose=None, init_bone=None):
    '''
    Initial fit with geometry: triangulation + Procrustes
    Input:
        bird: bird model
        keypoints (vn, kn, 3): 2d keypoints from each view with hard confidence
        frames (vn): frame index number
    '''
    
    # 3D kpts on bird
    if init_pose==None and init_bone==None:
        bird_kpts = torch.matmul(bird.vert2kpt, bird.V[0])
    else:
        init_ori = torch.zeros([1, 3]).float()
        
        bird_output = bird(init_ori, init_pose, init_bone)
        bird_kpts = bird_output['keypoints'][0]
        
    # Triangulation with LBFGS
    kpts_3d = mutils.get_gt_3d(keypoints, frames, LBFGS=True)
    
    valid_3d = kpts_3d[:, -1]>0
    valid_kpts_3d = kpts_3d[valid_3d, :3]
    bird_kpts = bird_kpts[valid_3d, :]
    
    # Procrustes with available 3D kpts
    R, t, s = mutils.Procrustes(bird_kpts, valid_kpts_3d)
    aa, _ = cv2.Rodrigues(R.numpy())

    init_ori = torch.tensor(aa).reshape(1,3).float()
    init_t = t
    init_s = s

    return init_ori, init_t, init_s
    
    
def fit_mesh(bird, optimizer, keypoints, frames, device='cpu',
            init_pose=None, init_bone=None):
    
    '''
    Only used in multiview and crossview fitting:
    Input:
        init_pose (vn, 24*3): body pose in axis-angle (exclude root joint orient)
        init_bone (vn, 24): bone length
    '''
    
    ### Triangulation + Procrustes as initialization
    init_ori, init_t, init_s = fit_geometry(bird, keypoints, frames, 
                                        init_pose=init_pose, init_bone=init_bone)

    ### If not provided (as in multiview), initialize with canonical
    if init_pose is None:
        init_pose = torch.zeros([1, 24*3])
    if init_bone is None:
        init_bone = torch.ones([1, 24])

    #### Change suitable format for optimizer
    ###### particularly, combine orient and body pose
    init_pose = torch.cat([init_ori, init_pose], dim=1)
    init_pose = init_pose.float().to(device)
    init_bone = init_bone.float().to(device)
    init_s = init_s.float().to(device)
    init_t = init_t.float().to(device)
    
    ### Camera parameter
    cam_rot, cam_t, focal, center, distortion = mutils.get_cam(frames)
    cam_rot = cam_rot.to(device)
    cam_t = cam_t.to(device)
    focal = focal.to(device)
    center = center.to(device)
    distortion = distortion.to(device)
    
    ### Mesh fitting
    keypoints = keypoints.to(device)
    vertices, pose_est, bone_est, scale_est, t = optimizer(init_pose, init_bone, 
                                                            init_t, init_s, 
                                                            cam_rot, cam_t, focal, center, 
                                                            keypoints, distortion=distortion)
    
    ### Generating mesh output
    bird_output = bird(pose_est[:, 0:3], 
                       pose_est[:, 3:], 
                       bone_est, scale_est)
    
    vertex_posed = bird_output['vertices'] + t
    mesh_keypoint = bird_output['keypoints'] + t
    
    return vertex_posed, mesh_keypoint, t, pose_est, bone_est, scale_est


def multiview_rigid_alignment(bird, pose, bone, keypoints, frames, device='cpu', num_iters=100):
    '''
    Rigidly align single view reconstruction to multiview instance so we can 
    check reconstruction accuracy across different views.
    1. First run general Procrustes for global alignment
    2. Because Procrustes can only use keypoints that are visible from at least two views, 
    we run a short optimizition (rigidly, fixed pose and shape) afterward to improve alignment. 
    Input:
        pose and bone are from singleview reconstruction;
        keypoints are ground truth for alignments
    '''
    
    ### Camera parameter
    cam_rot, cam_t, focal, center, distortion = mutils.get_cam(frames, device)
    
    ### Triangulation + Procrustes for global alignemnt
    global_orient, global_t, scale = fit_geometry(bird, keypoints, frames, 
                                            init_pose=pose, init_bone=bone)
    
    ### Optimization to improve alignment
    pose = pose.detach().clone().to(device)
    bone = bone.detach().clone().to(device)
    keypoints = keypoints.clone().to(device)
    batch_size = len(frames)

    global_orient = global_orient.to(device)
    global_t = global_t.to(device)
    scale = scale.to(device)

    global_orient.requires_grad=True
    global_t.requires_grad = True
    scale.requires_grad = True
    
    global_params = [global_orient, global_t, scale]
    global_optimizer = torch.optim.Adam(global_params, lr=1e-2, betas=(0.9, 0.999))
    for i in range(num_iters):
        bird_output = bird(global_pose=global_orient,
                           body_pose=pose,
                           bone_length=bone,
                           scale=scale)
        
        model_keypoints = bird_output['keypoints'] + global_t.repeat(1, 1, 1)
        model_keypoints = model_keypoints.repeat([batch_size, 1, 1])

        loss = camera_fitting_loss(model_keypoints, cam_rot, cam_t, focal, center,
                                    keypoints[:, :, :2], keypoints[:, :, -1], distortion=distortion)

        global_optimizer.zero_grad()
        loss.backward()
        global_optimizer.step()
        
    # Output
    bird_output = bird(global_pose=global_orient,
                       body_pose=pose,
                       bone_length=bone,
                       scale=scale)
    model_mesh = bird_output['vertices'] + global_t.repeat(1, 1, 1)
    model_keypoints = bird_output['keypoints'] + global_t.repeat(1, 1, 1)

    model_mesh = model_mesh.detach().to('cpu')
    model_keypoints = model_keypoints.detach().to('cpu')
    
    return model_mesh, model_keypoints

    
def reproject_masks(vertex_est, renderer_list, frames):
    
    # Transform vertex for each camera view
    cam_rot, cam_t, focal, center, distortion = mutils.get_cam(frames)
    rotation = cam_rot
    translation = cam_t.unsqueeze(1)

    points = vertex_est.repeat([len(frames),1,1])
    points = torch.einsum('bij,bkj->bki', rotation, points) + translation

    # Apply Distortion
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
    
    # Render for each view
    img = torch.zeros([1200, 1920, 3])
    proj_masks = []
    for i in range(len(frames)):
        renderer = renderer_list[frames[i]]
        img_pose, depth_map = renderer(points[i].cpu().numpy(), np.eye(3), [0,0,0], img.clone().numpy())
        mask = torch.tensor(depth_map>0)
        proj_masks.append(mask)
    
    return proj_masks


def reproject_keypoints(mesh_keypoints, frames):
    
    cam_rot, cam_t, focal, center, distortion = mutils.get_cam(frames)

    kpts = mesh_keypoints.repeat([len(frames), 1, 1])
    proj_kpts = perspective_projection(kpts, cam_rot, cam_t, focal, center, distortion)
    
    return proj_kpts

