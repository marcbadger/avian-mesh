import os
import json
import torch
from .LBS import LBS

class bird_model():
    '''
    Implementation of skined linear bird model
    '''
    def __init__(self, device=torch.device('cpu'), mesh='bird_eccv.json'):

        self.device = device

        # read in bird_mesh from the same dir
        this_dir = os.path.dirname(__file__)
        mesh_file = os.path.join(this_dir, mesh)
        with open(mesh_file, 'r') as infile:
            dd = json.load(infile)
        
        self.dd = dd
        self.kintree_table = torch.tensor(dd['kintree_table']).to(device)
        self.parents = self.kintree_table[0]
        self.weights = torch.tensor(dd['weights']).to(device)
        self.vert2kpt = torch.tensor(dd['vert2kpt']).to(device)

        self.J = torch.tensor(dd['J']).unsqueeze(0).to(device)
        self.V = torch.tensor(dd['V']).unsqueeze(0).to(device)
        self.LBS = LBS(self.J, self.parents, self.weights)

        prior = torch.load(this_dir + '/pose_bone_prior.pth')
        self.p_m = prior['p_m'].to(device)
        self.b_m = prior['b_m'].to(device)
        self.p_cov = prior['p_cov'].to(device)
        self.b_cov = prior['b_cov'].to(device)
        
    def __call__(self, global_pose, body_pose, bone_length, scale=1, pose2rot=True):
        batch_size = global_pose.shape[0]
        V = self.V.repeat([batch_size, 1, 1]) * scale

        # concatenate bone and pose
        bone = torch.cat([torch.ones([batch_size,1]).to(self.device), bone_length], dim=1)
        pose = torch.cat([global_pose, body_pose], dim=1)
        
        # LBS          
        verts = self.LBS(V, pose, bone, scale, to_rotmats=pose2rot)

        # Calculate 3d keypoint from new vertices resulted from pose
        keypoints = []
        for i in range(verts.shape[0]):
            kpt = torch.matmul(self.vert2kpt, verts[i])
            keypoints.append(kpt)
        keypoints = torch.stack(keypoints)

        # Final output after articulation
        output = {'vertices': verts,
                  'keypoints': keypoints}

        return output


