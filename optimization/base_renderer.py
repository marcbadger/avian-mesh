import numpy as np
import torch

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, SoftSilhouetteShader,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams
)

class base_renderer():
    def __init__(self, size, focal=None, fov=None, device='cpu'):
        self.device = device
        self.size = size
        self.camera = self.init_camera(focal, fov)
        self.silhouette_renderer = self.init_silhouette_renderer()

        self.R = torch.tensor([[-1, 0, 0],
                               [0, -1, 0],
                               [0, 0, 1]]).repeat(1,1,1).to(device)
        self.T = torch.zeros(3).repeat(1, 1).to(device)

    def init_camera(self, focal, fov):
        if fov is None:
            fov = 2 * np.arctan (self.size/(focal * 2)) * 180 / np.pi
            
        camera = OpenGLPerspectiveCameras(zfar=350, fov=fov, device=self.device)
        return camera

    def init_silhouette_renderer(self):
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size  = self.size, 
            blur_radius = np.log(1./1e-4 - 1.) * blend_params.sigma, 
            faces_per_pixel=100)

        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.camera, 
                raster_settings=raster_settings
                ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )
        return silhouette_renderer

    def __call__(self, vertices, faces):
        ''' Right now only render silhouettes
            Input: 
            vertices: BN * V * 3
            faces: BN * F * 3
        '''
        torch_mesh = Meshes(verts=vertices.to(self.device), 
                            faces=faces.to(self.device))
        silhouette = self.silhouette_renderer(meshes_world=torch_mesh.clone(), 
                                              R=self.R, T=self.T)

        return silhouette


