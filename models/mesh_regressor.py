import os
import numpy as np
import torch
import torch.nn as nn

from utils.geometry import rot6d_to_rotmat

'''
Implementation of the mesh regressor
'''

class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        self.layer1 = self.make_layer(36, 512)
        self.layer2 = self.make_layer(512, 512)
        self.final  = nn.Linear(512, 25*6+3)
        
    def make_layer(self, in_channel, out_channel):
        modules = [ nn.Linear(in_channel, out_channel),
                    nn.BatchNorm1d(out_channel),
                    nn.ReLU() ]

        return nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.final(x)
        return x
    

class ShapeNet(nn.Module):
    def __init__(self):
        super(ShapeNet, self).__init__()
        self.layer1 = self.make_layer(1, 8, 5, 2, 2)
        self.layer2 = self.make_layer(8, 8, 3, 1, 1)
        self.layer3 = self.make_layer(8, 16, 3, 1, 1)
        self.layer4 = self.make_layer(16, 32, 3, 1, 1)
        self.layer5 = self.make_layer(32, 64, 3, 1, 1)
        
        self.fc = nn.Linear(1024, 24)
    
    def make_layer(self, in_channel, out_channel, k, s, p):
        modules = [ nn.Conv2d(in_channel, out_channel, k, s, p),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2,0)
                  ]
        
        return nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        
        return x
    

class DigiNet(nn.Module):
    def __init__(self):
        super(DigiNet, self).__init__()
        self.posenet = PoseNet()
        self.shapenet = ShapeNet()
    
    def forward(self, kpts, masks):
        pose_tran = self.posenet(kpts)
        bone = self.shapenet(masks)
        return pose_tran, bone

    def postprocess(self, p_est, b_est):
        """
        Convert 6d rotation to 9d rotation
        Input:
            p_est: pose_tran from forward()
            b_est: bone from forward()
        """
        pose_6d = p_est[:, :-3].contiguous()
        p_est_rot = rot6d_to_rotmat(pose_6d).view(-1, 25*9)
        
        pose = p_est_rot
        tran = p_est[:, -3:]
        bone = b_est

        return pose, tran, bone


def mesh_regressor():
    model = DigiNet()
    return model


def load_regressor(device='cpu'):
    this_dir = os.path.dirname(__file__)
    checkpoint = torch.load(this_dir+'/regressor.pth', map_location=device)

    model = mesh_regressor()

    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model

