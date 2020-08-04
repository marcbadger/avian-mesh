import torch
import os

from models import bird_model
from .losses import camera_fitting_loss, body_fitting_loss


class OptimizeMV():
    '''
    Implementation of multiview reconstruction optimizer
    This version forgos using silhouttes for speedy reconstruction
    '''
    def __init__(self, lim_weight=1, prior_weight=1, bone_weight=1,
                 step_size=1e-2,
                 num_iters=100,
                 device=torch.device('cpu'), mesh='bird_eccv.json'):

        # Store optimization hyperparameters
        self.device = device
        self.step_size = step_size
        self.num_iters = num_iters
        self.lim_weight = lim_weight
        self.prior_weight = prior_weight
        self.bone_weight = bone_weight

        # Load Bird Mesh Model
        self.bird = bird_model(device=device, mesh=mesh)
        self.faces = torch.tensor(self.bird.dd['F'])



    def __call__(self, init_pose, init_bone, init_t, scale, cam_rot, cam_t, focal_length, camera_center, 
                keypoints, distortion=None):
        """Perform multiview reconstruction
        Input:
            model:
            init_pose: (1, 25*3) initial pose estimate
            init_bone: (1, 24) initial bone estimate
            init_t: (1, 3) initial translation estimate
            scale: (1,) initial scale estimate
            
            multiview:
            cam_rot: (VN, 3, 3) multiview camera orientations, VN is number of views
            cam_t: (VN, 3) camera translations
            focal: (VN,) camera focal length
            cam_center: (VN, 2) camera center
            distortion: (VN, 5) camera distortion factor, if provided
            keypoints: (VN, 12, 3) keypoints with confidence, seen from multiple views
            
        """

        # Number of views
        batch_size = cam_rot.shape[0]

        # Unbind keypoint location and confidence
        keypoints_2d = keypoints[:, :, :2]
        keypoints_conf = keypoints[:, :, -1]

        # Copy all initialization
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        global_t = init_t.detach().clone()
        bone_length = init_bone.detach().clone()
        scale = scale.detach().clone()

        # Step 1: Optimize global orientation, translation and scale
        body_pose.requires_grad=False
        bone_length.requires_grad=False
        global_orient.requires_grad=True
        global_t.requires_grad = True
        scale.requires_grad = True

        gloabl_opt_params = [global_orient, global_t, scale]
        gloabl_optimizer = torch.optim.Adam(gloabl_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        for i in range(self.num_iters):
            bird_output = self.bird(global_pose=global_orient,
                                    body_pose=body_pose,
                                    bone_length=bone_length,
                                    scale=scale)
            model_keypoints = bird_output['keypoints'] + global_t.repeat(1, 1, 1)
            model_keypoints = model_keypoints.repeat([batch_size, 1, 1])

            loss = camera_fitting_loss(model_keypoints, cam_rot, cam_t, focal_length, camera_center,
                                       keypoints_2d, keypoints_conf, distortion=distortion)

            gloabl_optimizer.zero_grad()
            loss.backward()
            gloabl_optimizer.step()


        # Step 2: Optimize all parameters        
        body_pose.requires_grad=True
        bone_length.requires_grad=True
        global_orient.requires_grad=True
        global_t.requires_grad = True
        scale.requires_grad = True

        body_opt_params = [body_pose, bone_length, global_orient, global_t, scale]
        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        ### disable wing tip at this stage
        kpts_conf = keypoints_conf.clone()
        kpts_conf[:, 5:7] = 0

        for i in range(self.num_iters):
            bird_output = self.bird(global_pose=global_orient,
                                    body_pose=body_pose,
                                    bone_length=bone_length,
                                    scale=scale)
            model_keypoints = bird_output['keypoints'] + global_t.repeat(1, 1, 1)
            model_keypoints = model_keypoints.repeat([batch_size, 1, 1])

            loss = body_fitting_loss(model_keypoints, cam_rot, cam_t, focal_length, camera_center,
                                     keypoints_2d, kpts_conf, body_pose, bone_length, 
                                     lim_weight=self.lim_weight, prior_weight=self.prior_weight, 
                                     bone_weight=self.bone_weight, distortion=distortion)


            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()


        # Step 3: Refinement: Wing tips
        body_pose.requires_grad=True
        bone_length.requires_grad=True
        global_orient.requires_grad= True
        global_t.requires_grad = True
        scale.requires_grad = True

        body_opt_params = [body_pose, bone_length, global_orient, global_t, scale]
        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))
        
        pose_init = body_pose.detach().clone()
        bone_init = bone_length.detach().clone()

        for i in range(100):
            bird_output = self.bird(global_pose=global_orient,
                                    body_pose=body_pose,
                                    bone_length=bone_length,
                                    scale=scale)
            model_keypoints = bird_output['keypoints'] + global_t.repeat(1, 1, 1)
            model_keypoints = model_keypoints.repeat([batch_size, 1, 1])

            loss = body_fitting_loss(model_keypoints, cam_rot, cam_t, focal_length, camera_center,
                                    keypoints_2d, keypoints_conf, body_pose, bone_length, sigma=100,
                                    lim_weight=self.lim_weight, prior_weight=self.prior_weight, 
                                    bone_weight=self.bone_weight, distortion=distortion,
                                    pose_init=pose_init, bone_init=bone_init)


            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()


        # Output
        vertices = bird_output['vertices'].detach().cpu()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach().cpu()
        bone = bone_length.detach().cpu()
        scale = scale.detach().cpu()
        global_t = global_t.detach().cpu()

        return vertices, pose, bone, scale, global_t


