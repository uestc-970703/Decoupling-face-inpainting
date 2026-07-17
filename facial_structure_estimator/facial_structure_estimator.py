import os
import numpy as np
import torch
import torch.nn as nn

from demo import networks
from lap.renderer.renderer_mr import Renderer
from demo.utils import get_grid


EPS = 1e-7


class FacialStructureEstimator(nn.Module):
    """Frozen LAP estimator used as a differentiable training-time supervisor."""

    def __init__(self, checkpoint_lap, output_size=128, device='cuda:0'):
        super().__init__()
        self.device_name = str(device)
        self.checkpoint_lap = checkpoint_lap
        self.output_size = output_size
        self.image_size_lap = 128
        self.min_depth = 0.9
        self.max_depth = 1.1
        self.border_depth = 1.05
        self.xyz_rotation_range = 60
        self.xy_translation_range = 0.1
        self.z_translation_range = 0
        self.fov = 10
        self.inner_batch = 1
        self.count = 0

        self.depth_rescaler = lambda d: (1 + d) / 2 * self.max_depth + (1 - d) / 2 * self.min_depth

        fx = (self.image_size_lap - 1) / 2 / (np.tan(self.fov / 2 * np.pi / 180))
        fy = (self.image_size_lap - 1) / 2 / (np.tan(self.fov / 2 * np.pi / 180))
        cx = (self.image_size_lap - 1) / 2
        cy = (self.image_size_lap - 1) / 2
        k = torch.FloatTensor([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
        self.register_buffer('inv_K_lap', torch.inverse(k).unsqueeze(0))

        self.renderer_mr = Renderer({
            'device': self.device_name,
            'min_depth': self.min_depth,
            'max_depth': self.max_depth,
            'fov': self.fov,
        }, im_size=128)

        self.netD = networks.ED_Aggregation(cin=3, cout=1, nf=64, zdim=512, activation=None, inner_batch=self.inner_batch, count=self.count)
        self.netA = networks.ED_Aggregation(cin=3, cout=3, nf=64, zdim=512, inner_batch=self.inner_batch, count=self.count)
        self.refine_netA = networks.ED_attribute_refining(cin=3, cout=3, nf=64, zdim=512)
        self.refine_netD = networks.ED_attribute_refining(cin=1, cout=1, nf=64, zdim=512, activation=None)
        self.netL = networks.Encoder(cin=3, cout=4, nf=32)
        self.netV = networks.Encoder(cin=3, cout=6, nf=32)

        self.load_checkpoint()
        self.eval()
        self.requires_grad_(False)

    def load_checkpoint(self):
        checkpoint_path = os.path.abspath(self.checkpoint_lap)
        cp_lap = torch.load(checkpoint_path, map_location='cpu')
        self.netD.load_state_dict(cp_lap['netD'])
        self.netA.load_state_dict(cp_lap['netA'])
        self.refine_netD.load_state_dict(cp_lap['refine_netD'])
        self.refine_netA.load_state_dict(cp_lap['refine_netA'])
        self.netL.load_state_dict(cp_lap['netL'])
        self.netV.load_state_dict(cp_lap['netV'])

    def depth_to_3d_grid(self, depth, inv_K):
        b, h, w = depth.shape
        grid_2d = get_grid(b, h, w, normalize=False).to(depth.device)
        depth = depth.unsqueeze(-1)
        grid_3d = torch.cat((grid_2d, torch.ones_like(depth)), dim=3)
        return grid_3d.matmul(inv_K.to(depth.device).transpose(2, 1)) * depth

    def get_normal_from_depth(self, depth, inv_K):
        b, h, w = depth.shape
        grid_3d = self.depth_to_3d_grid(depth, inv_K)
        tu = grid_3d[:, 1:-1, 2:] - grid_3d[:, 1:-1, :-2]
        tv = grid_3d[:, 2:, 1:-1] - grid_3d[:, :-2, 1:-1]
        normal = tu.cross(tv, dim=3)
        zero = normal.new_tensor([0, 0, 1])
        normal = torch.cat([zero.repeat(b, h - 2, 1, 1), normal, zero.repeat(b, h - 2, 1, 1)], 2)
        normal = torch.cat([zero.repeat(b, 1, w, 1), normal, zero.repeat(b, 1, w, 1)], 1)
        return normal / (((normal ** 2).sum(3, keepdim=True)) ** 0.5 + EPS)

    def forward(self, image):
        """Estimate LAP supervision maps from an image tensor in [-1, 1]."""
        image_01 = (image.clamp(-1, 1) + 1) / 2
        input_im_lap = nn.functional.interpolate(image_01, (self.image_size_lap, self.image_size_lap), mode='bilinear', align_corners=False) * 2 - 1
        b, _, h_lap, w_lap = input_im_lap.shape
        input_half = nn.functional.interpolate(input_im_lap, (128, 128), mode='bilinear', align_corners=False)

        canon_depth_raw = self.netD(input_im_lap, 1)
        canon_depth_raw = self.refine_netD(canon_depth_raw, input_half).squeeze(1)
        canon_depth_lap = canon_depth_raw - canon_depth_raw.view(b, -1).mean(1).view(b, 1, 1)
        canon_depth_lap = self.depth_rescaler(canon_depth_lap.tanh())

        depth_border_lap = torch.zeros(1, h_lap, w_lap - 4, device=input_im_lap.device)
        depth_border_lap = nn.functional.pad(depth_border_lap, (2, 2), mode='constant', value=1)
        canon_depth_lap = canon_depth_lap * (1 - depth_border_lap) + depth_border_lap * self.border_depth
        canon_depth_lap = canon_depth_lap.clone()
        canon_depth_lap[:, -20:, :15] = self.border_depth

        canon_albedo_lap = self.netA(input_im_lap, 1)
        canon_albedo_lap = self.refine_netA(canon_albedo_lap, input_half)

        canon_light = self.netL(input_im_lap)
        canon_light_a_lap = canon_light[:, :1] / 2 + 0.5
        canon_light_b_lap = canon_light[:, 1:2] / 2 + 0.5
        canon_light_d_lap = torch.cat([canon_light[:, 2:], torch.ones(b, 1, device=input_im_lap.device)], 1)
        canon_light_d_lap = canon_light_d_lap / ((canon_light_d_lap ** 2).sum(1, keepdim=True)) ** 0.5

        canonical_normal_lap = self.get_normal_from_depth(canon_depth_lap, inv_K=self.inv_K_lap)
        canon_diffuse_shading_lap = (canonical_normal_lap * canon_light_d_lap.view(-1, 1, 1, 3)).sum(3).clamp(min=0).unsqueeze(1)
        canon_shading = canon_light_a_lap.view(-1, 1, 1, 1) + canon_light_b_lap.view(-1, 1, 1, 1) * canon_diffuse_shading_lap
        canonical_image_lap = (canon_albedo_lap / 2 + 0.5) * canon_shading * 2 - 1

        view_lap = self.netV(input_im_lap)
        view_lap = torch.cat([
            view_lap[:, :3] * np.pi / 180 * self.xyz_rotation_range,
            view_lap[:, 3:5] * self.xy_translation_range,
            view_lap[:, 5:] * self.z_translation_range], 1)

        self.renderer_mr.set_transform_matrices(view_lap)
        recon_depth = self.renderer_mr.warp_canon_depth(canon_depth_lap)
        recon_normal_lap = self.renderer_mr.get_normal_from_depth(recon_depth)

        canonical_image_lap = nn.functional.interpolate(canonical_image_lap, (self.output_size, self.output_size), mode='bilinear', align_corners=False)
        canonical_normal_lap = nn.functional.interpolate(canonical_normal_lap.permute(0, 3, 1, 2), (self.output_size, self.output_size), mode='bilinear', align_corners=False)
        recon_normal_lap = nn.functional.interpolate(recon_normal_lap.permute(0, 3, 1, 2), (self.output_size, self.output_size), mode='bilinear', align_corners=False)
        canonical_normal_lap = canonical_normal_lap / ((canonical_normal_lap ** 2).sum(1, keepdim=True) ** 0.5 + EPS)
        recon_normal_lap = recon_normal_lap / ((recon_normal_lap ** 2).sum(1, keepdim=True) ** 0.5 + EPS)

        return {
            'canonical_image_lap': canonical_image_lap,
            'canonical_normal_lap': canonical_normal_lap,
            'recon_normal_lap': recon_normal_lap,
        }
