import os
#os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh

class Renderer:
    """
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, center=None, img_w=None, img_h=None, faces=None):

        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w,
                                       viewport_height=img_h,
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = center
        self.faces = faces

    def __call__(self, vertices, cam_rot, cam_t, image):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.8, 0.3, 0.3, 1.0))


        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, :3] = cam_rot
        camera_pose[:3, 3] = cam_t
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1],
                                           zfar=1000)
        scene.add(camera, pose=camera_pose)


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.uint8)
        valid_mask = (rend_depth>0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img, rend_depth


class Silhouette_Renderer:

    def __init__(self, focal_length=5000, center=None, img_w=None, img_h=None):

        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w,
                                       viewport_height=img_h,
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = center
        self.camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                                cx=self.camera_center[0], cy=self.camera_center[1],
                                                zfar=1000)
        self.camera_pose = np.eye(4)
        self.camera_pose[:3, :3] = np.eye(3)

    def __call__(self, vertices, faces):

        mesh = trimesh.Trimesh(vertices, faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')
        scene.add(self.camera, pose=self.camera_pose)

        rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        silhouette = rend_depth > 0
        
        return silhouette
