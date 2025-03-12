from typing import Any, Dict, List, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
import numpy as np

from diffusers.models.attention_processor import Attention
from diffusers.models.attention import BasicTransformerBlock, AdaLayerNorm
from diffusers.models.controlnet import zero_module
from .embedder import get_embedder


def _ensure_kv_is_int(view_pair: dict):
    """yaml key can be int, while json cannot. We convert here.
    """
    new_dict = {}
    for k, v in view_pair.items():
        new_value = [int(vi) for vi in v]
        new_dict[int(k)] = new_value
    return new_dict


class GatedConnector(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        data = torch.zeros(dim)
        self.alpha = nn.parameter.Parameter(data)

    def forward(self, inx):
        # as long as last dim of input == dim, pytorch can auto-broad
        return F.tanh(self.alpha) * inx


class MultiModalityTransformerBlock(nn.Module):

    def __init__(
        self,
        dim_pc: int,
        dim_img: int,
        num_attention_heads: int,
        attention_head_dim: int,
        img_scale_factor: int,
        query_pos_embed=True,
        fov=(-30, 10),
        pc_ray_sample_num=32,
        img_ray_sample_num=32,
        dropout=0.0,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
        # multi_view
        zero_module_type="zero_linear",
        image_size=(400, 224),

        pc_max_range = 80,
        img_max_depth = 80,

        num_freqs=8,
        depth_embed=None,
    ):
        super().__init__()

        self.dim_pc = dim_pc
        self.dim_img = dim_img
        # self.img_scale_factor = img_scale_factor
        self.pc_ray_sample_num = pc_ray_sample_num
        self.img_ray_sample_num = img_ray_sample_num
        self.fov = (fov[0]/180*math.pi, fov[1]/180*math.pi)
        self.use_ada_layer_norm = norm_type == "ada_norm"
        self.image_size = image_size

        self.pc_max_range = pc_max_range
        self.img_max_depth = img_max_depth

        self.query_pos_embed = query_pos_embed
        if query_pos_embed:
            if self.query_pos_embed == 'fourier':
                self.fourier_embedder_pc = get_embedder(3, num_freqs)
                pc_pos_dim = self.fourier_embedder_pc.out_dim
                self.fourier_embedder_img = get_embedder(6, num_freqs)
                img_pos_dim = self.fourier_embedder_img.out_dim
            else:
                pc_pos_dim = 3
                img_pos_dim = 6

            self.pc_pos_embed = nn.Sequential(
                nn.Linear(pc_pos_dim, self.dim_pc),
                nn.ReLU(),
                zero_module(nn.Linear(self.dim_pc, self.dim_pc))
            )

            self.img_pos_embed = nn.Sequential(
                nn.Linear(img_pos_dim, self.dim_img),
                nn.ReLU(),
                zero_module(nn.Linear(self.dim_img, self.dim_img))
            )

        self.depth_embed = depth_embed
        if depth_embed:
            if depth_embed == "fourier":
                self.fourier_embedder_depth = get_embedder(1, num_freqs)
                depth_in_dim = self.fourier_embedder_depth.out_dim
            else:
                depth_in_dim = 1
            self.pc_depth_pos_embed = nn.Sequential(
                nn.Linear(depth_in_dim, self.dim_img),
                nn.ReLU(),
                nn.Linear(self.dim_img, self.dim_img)
            )
            self.img_depth_pos_embed = nn.Sequential(
                nn.Linear(depth_in_dim, self.dim_pc),
                nn.ReLU(),
                nn.Linear(self.dim_pc, self.dim_pc)
            )

        # multiview attention
        self.norm1 = (
            AdaLayerNorm(dim_pc, num_embeds_ada_norm)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim_pc, elementwise_affine=norm_elementwise_affine)
        )
        self.norm2 = (
            AdaLayerNorm(dim_img, num_embeds_ada_norm)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim_img, elementwise_affine=norm_elementwise_affine)
        )
        self.attn1 = Attention(
            query_dim=dim_pc,
            cross_attention_dim=dim_img,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        self.attn2 = Attention(
            query_dim=dim_img,
            cross_attention_dim=dim_pc,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        if zero_module_type == "zero_linear":
            # NOTE: zero_module cannot apply to successive layers.
            self.connector1 = zero_module(nn.Linear(dim_pc, dim_pc))
            self.connector2 = zero_module(nn.Linear(dim_img, dim_img))

        elif zero_module_type == "gated":
            self.connector1 = GatedConnector(dim_pc)
            self.connector2 = GatedConnector(dim_img)

        elif zero_module_type == "none":
            # TODO: if this block is in controlnet, we may not need zero here.
            self.connector1 = lambda x: x
            self.connector2 = lambda x: x

        else:
            raise TypeError(f"Unknown zero module type: {zero_module_type}")

    @property
    def new_module(self):
        ret = {
            "norm1": self.norm1,
            "norm2": self.norm2,
            "attn1": self.attn1,
            "attn2": self.attn2,
        }
        if isinstance(self.connector1, nn.Module):
            ret["connector1"] = self.connector1
        if isinstance(self.connector2, nn.Module):
            ret["connector2"] = self.connector2
        if self.query_pos_embed:
            ret["pc_pos"] = self.pc_pos_embed
            ret["img_pos"] = self.img_pos_embed
        if self.depth_embed:
            ret["pc_depth"] = self.pc_depth_pos_embed
            ret["img_depth"] = self.img_depth_pos_embed           
        return ret

    def _img_sample_pts_along_rays(self, lidar2imgs, height, width):
        # sample_depths = torch.linspace(1, 60, self.img_ray_sample_num).type_as(lidar2imgs)
        index = torch.arange(start=0, end=self.img_ray_sample_num, step=1, device=lidar2imgs.device).float()
        index_1 = index + 1
        bin_size = (self.img_max_depth - 1) / (self.img_ray_sample_num * (1 + self.img_ray_sample_num))
        sample_depths = 1 + bin_size * index * index_1

        xs = np.arange(width) + 0.5
        ys = np.arange(height) + 0.5
        
        xs = xs * (self.image_size[0] / width)
        ys = ys * (self.image_size[1] / height)

        xs, ys = np.meshgrid(xs, ys, indexing='xy')

        coords_img_2d = np.stack([xs, ys], axis=-1)  # [height, width, 2]
        coords_img_2d = torch.from_numpy(coords_img_2d).type_as(sample_depths)  # [height, width, 2]
        coords_img_2d = coords_img_2d[:, :, None].repeat(1, 1, self.img_ray_sample_num, 1)  # [height, width, sample_num, 2]
        sample_depths = sample_depths[None, None].repeat(height, width, 1)
        coords_img_3d = torch.cat([coords_img_2d, sample_depths[..., None]], dim=-1)  # [height, width, sample_num, 3]

        coords_img_3d[..., :2] = coords_img_3d[..., :2] * coords_img_3d[..., 2:3]
        coords_img_4d = torch.cat([coords_img_3d, torch.ones_like(coords_img_3d[..., :1])], dim=-1)  # [height, width, sample_num, 4]

        coords_img_4d = coords_img_4d[None, None, :, :, :, :, None]  # [1, 1, height, width, sample_num, 4, 1]
        coords_4d = torch.matmul(torch.inverse(lidar2imgs[:, :, None, None, None].float()).type_as(coords_img_4d), coords_img_4d).squeeze(-1)  # [bs, num_views, height, width, sample_num, 4]

        coords_3d = coords_4d[..., :3]  # [bs, num_views, height, width, sample_num, 3]

        return coords_3d


    def _pc_sample_pts_along_rays(self, lidar2imgs, height, width):
        device = lidar2imgs.device
        # sample_depths = torch.linspace(1, 60, self.pc_ray_sample_num).to(device)
        index = torch.arange(start=0, end=self.pc_ray_sample_num, step=1, device=device).float()
        index_1 = index + 1
        bin_size = (self.pc_max_range - 1) / (self.pc_ray_sample_num * (1 + self.pc_ray_sample_num))
        sample_depths = 1 + bin_size * index * index_1

        yaw = (width - 0.5 - torch.arange(0, width)) / width * 2. * torch.pi - torch.pi


        # yaw = (torch.arange(width) + 0.5) / width * 2 * math.pi
        pitch = self.fov[1] - (torch.arange(height) + 0.5) / height * (self.fov[1] - self.fov[0])

        yaw = yaw.to(device)
        pitch = pitch.to(device)

        zs = torch.sin(pitch[:, None]).repeat(1, yaw.shape[0])
        ys = torch.cos(pitch[:, None]) * torch.sin(yaw[None])
        xs = torch.cos(pitch[:, None]) * torch.cos(yaw[None])

        coords_3d = torch.stack([xs, ys, zs], dim=-1)
        coords_3d = coords_3d[:, :, None] * sample_depths[None, None, :, None]

        return coords_3d

    def _epipolar_projection_pc(self, sample_pc, sample_img, lidar2imgs):
        pc_height, pc_width = sample_pc.shape[1:3]
        img_height, img_width = sample_img.shape[2:4]
        bs, n_cam = sample_img.shape[:2]
        coords_3d = self._pc_sample_pts_along_rays(lidar2imgs, pc_height, pc_width)
        coords_4d = torch.cat([coords_3d, torch.ones_like(coords_3d[..., :1])], dim=-1)
        coords_proj = lidar2imgs[:,:, None, None, None] @ coords_4d[None, None, ..., None]
        coords_proj = coords_proj.squeeze(-1)
        coords_proj[..., :2] = coords_proj[..., :2] / (coords_proj[..., 2:3] + 1e-5)
        # coords_proj[..., :2] = coords_proj[..., :2] / self.img_scale_factor
        coords_proj[..., 0] = coords_proj[..., 0] / (self.image_size[0] / img_width)
        coords_proj[..., 1] = coords_proj[..., 1] / (self.image_size[1] / img_height)

        proj_mask = (coords_proj[..., 0]>0) & (coords_proj[..., 0]<img_width) & (coords_proj[..., 1]>0) & (coords_proj[..., 1]<img_height) & (coords_proj[..., 2]>0)


        coords_proj = coords_proj[..., :2]

        coords_proj = coords_proj.flatten(2,3).flatten(0,1)
        coords_proj[..., 0] = coords_proj[..., 0] / img_width * 2 - 1
        coords_proj[..., 1] = coords_proj[..., 1] / img_height * 2 - 1

        proj_feat = F.grid_sample(sample_img.flatten(0, 1).permute(0, 3, 1, 2), coords_proj, align_corners=False)

        proj_feat = proj_feat.view(bs, n_cam, self.dim_img, pc_height, pc_width, self.pc_ray_sample_num)

        proj_feat = proj_feat * proj_mask[:, :, None]
        proj_feat = torch.sum(proj_feat, dim=1) / (torch.sum(proj_mask[:, :, None], dim=1) + 1e-5)

        proj_feat = proj_feat.permute(0, 2, 3, 4, 1)
        proj_mask = torch.sum(proj_mask, dim=1) > 0

        return proj_feat, proj_mask


    def _epipolar_projection_img(self, sample_img, sample_pc, lidar2imgs):
        pc_height, pc_width = sample_pc.shape[1:3]
        img_height, img_width = sample_img.shape[2:4]
        bs, n_cam = sample_img.shape[:2]
        coords_3d = self._img_sample_pts_along_rays(lidar2imgs, img_height, img_width)

        coords_r = torch.norm(coords_3d, dim=-1)
        pitch = torch.arcsin(coords_3d[..., 2] / coords_r)
        yaw = torch.atan2(coords_3d[..., 1], coords_3d[..., 0])

        # col_inds = self.width - 1.0 + 0.5 - (azi + np.pi) / (2.0 * np.pi) * self.width

        yaw = 1 - (yaw + math.pi) / (2.0 * math.pi)
        yaw = yaw * 2 - 1

        # yaw = torch.where(yaw>=0, yaw, yaw+2*math.pi)
        # yaw = yaw / (2* math.pi) * 2 - 1

        pitch = ((pitch - self.fov[0]) / (self.fov[1] - self.fov[0]) * 2 - 1) * -1

        coords_pc = torch.stack([yaw, pitch], dim=-1)
        proj_mask = (yaw > -1) & (yaw < 1) & (pitch > -1) & (pitch < 1)
        
        coords_pc = coords_pc.flatten(1, 3)
        proj_feat = F.grid_sample(sample_pc.permute(0, 3, 1, 2), coords_pc, align_corners=False)
        proj_feat = proj_feat.view(bs, self.dim_pc, n_cam, img_height, img_width, self.img_ray_sample_num)

        proj_feat = proj_feat * proj_mask[:, None]
        proj_feat = proj_feat.permute(0, 2, 3, 4, 5, 1)

        return proj_feat, proj_mask

    def forward(
        self,
        sample_pc,
        sample_img,
        lidar2imgs,
        timestep=None,
    ):

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        bs = sample_pc.shape[0]
        n_cam = lidar2imgs.shape[1]
        pc_h, pc_w = sample_pc.shape[-2:]
        img_h, img_w = sample_img.shape[-2:]

        sample_pc = sample_pc.permute(0, 2, 3, 1)
        sample_img = sample_img.permute(0, 2, 3, 1)
        sample_pc = sample_pc.view(bs, pc_h, pc_w, self.dim_pc)
        sample_img = sample_img.view(bs, n_cam, img_h, img_w, self.dim_img)

        if self.query_pos_embed:
            pc_coords_3d_ = self._pc_sample_pts_along_rays(lidar2imgs, pc_h, pc_w)
            pc_coords_3d = pc_coords_3d_[..., 0, :]
            pc_coords_3d = pc_coords_3d / torch.norm(pc_coords_3d, dim=-1, keepdim=True)

            img_coords_3d_ = self._img_sample_pts_along_rays(lidar2imgs, img_h, img_w)
            img_coords_3d = img_coords_3d_[..., 0, :]

            cam_coords = torch.matmul(torch.inverse(lidar2imgs.float()), torch.Tensor([[[[self.image_size[0]/2], [self.image_size[1]/2], [0], [1]]]]).to(lidar2imgs.device))
            cam_coords = cam_coords.squeeze(-1)[..., :3]
            img_coords_3d = torch.cat([cam_coords[:, :, None, None].repeat(1, 1, img_h, img_w, 1), img_coords_3d], dim=-1)
  
            if self.query_pos_embed == 'fourier':
                pc_pos_3d = self.fourier_embedder_pc(pc_coords_3d / 2 + 0.5)
                img_coords_ray = img_coords_3d[..., 3:] / 2
                img_pos_3d = torch.cat([img_coords_3d[..., :3], img_coords_ray], dim=-1)
                img_pos_3d = self.fourier_embedder_img(img_pos_3d / 2 + 0.5)
            else:
                pc_pos_3d = pc_coords_3d
                img_pos_3d = img_coords_3d

            pc_pos_embed = self.pc_pos_embed(pc_pos_3d)

            sample_pc = sample_pc + pc_pos_embed[None]
            img_pos_embed = self.img_pos_embed(img_pos_3d)
            sample_img = sample_img + img_pos_embed
            

        sample_pc_ = sample_pc.clone()
        sample_img_ = sample_img.clone()

        if self.use_ada_layer_norm:
            sample_pc = self.norm1(sample_pc, timestep)
            sample_img = self.norm2(sample_img, timestep)

        else:
            sample_pc = self.norm1(sample_pc)
            sample_img = self.norm2(sample_img)

        pc_proj_features, pc_proj_mask = self._epipolar_projection_pc(sample_pc, sample_img, lidar2imgs)
        img_proj_features, img_proj_mask = self._epipolar_projection_img(sample_img, sample_pc, lidar2imgs)

        pc_proj_mask = pc_proj_mask.float()
        pc_proj_mask = torch.where(pc_proj_mask>0, 0.0, -1e4)

        img_proj_mask = img_proj_mask.float()
        img_proj_mask = torch.where(img_proj_mask>0, 0.0, -1e4)

        if self.depth_embed:
            pc_range = torch.norm(pc_coords_3d_, dim=-1, keepdim=True)
            if self.depth_embed == 'fourier':
                pc_range_embed = self.fourier_embedder_depth(pc_range / self.pc_max_range)
            else:
                pc_range_embed = pc_range
            pc_range_embed = self.pc_depth_pos_embed(pc_range_embed)
            pc_proj_features = pc_proj_features + pc_range_embed[None]

        attn_output = self.attn1(sample_pc.flatten(0,2)[:, None], encoder_hidden_states=pc_proj_features.flatten(0, 2), attention_mask=pc_proj_mask.flatten(0, 2)[:, None])
        attn_output = attn_output.view(bs, pc_h, pc_w, self.dim_pc)

        attn_output = self.connector1(attn_output)

        sample_pc = sample_pc_ + attn_output

        if self.depth_embed:
            index = torch.arange(start=0, end=self.img_ray_sample_num, step=1, device=lidar2imgs.device).float()
            index_1 = index + 1
            bin_size = (self.img_max_depth - 1) / (self.img_ray_sample_num * (1 + self.img_ray_sample_num))
            img_depth = 1 + bin_size * index * index_1
            img_depth = img_depth[..., None]
            if self.depth_embed == 'fourier':
                img_depth_embed = self.fourier_embedder_depth(img_depth / self.img_max_depth)
            else:
                img_depth_embed = img_depth
            img_depth_embed = self.img_depth_pos_embed(img_depth_embed)
            img_proj_features = img_proj_features + img_depth_embed[None, None, None, None]

        attn_output = self.attn2(sample_img.flatten(0,3)[:, None], encoder_hidden_states=img_proj_features.flatten(0, 3), attention_mask=img_proj_mask.flatten(0, 3)[:, None])
        attn_output = attn_output.view(bs, n_cam, img_h, img_w, self.dim_img)
        attn_output = self.connector2(attn_output)
        sample_img = sample_img_ + attn_output

        sample_pc = sample_pc.permute(0, 3, 1, 2)
        sample_img = sample_img.permute(0, 1, 4, 2, 3).flatten(0, 1)

        return sample_pc, sample_img


class RandomMultiModalityTransformerBlock(nn.Module):

    def __init__(
        self,
        dim_pc: int,
        dim_img: int,
        num_attention_heads: int,
        attention_head_dim: int,
        img_scale_factor: int,
        query_pos_embed=True,
        fov=(-30, 10),
        pc_ray_sample_num=32,
        img_ray_sample_num=32,
        dropout=0.0,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
        # multi_view
        zero_module_type="zero_linear",
        image_size=(400, 224),

        pc_max_range = 80,
        img_max_depth = 80,

        num_freqs=8,
        depth_embed=None,
    ):
        super().__init__()

        self.dim_pc = dim_pc
        self.dim_img = dim_img
        # self.img_scale_factor = img_scale_factor
        self.pc_ray_sample_num = pc_ray_sample_num
        self.img_ray_sample_num = img_ray_sample_num
        self.fov = (fov[0]/180*math.pi, fov[1]/180*math.pi)
        self.use_ada_layer_norm = norm_type == "ada_norm"
        self.image_size = image_size

        self.pc_max_range = pc_max_range
        self.img_max_depth = img_max_depth

        query_pos_embed = None
        self.query_pos_embed = query_pos_embed
        if query_pos_embed:
            if self.query_pos_embed == 'fourier':
                self.fourier_embedder_pc = get_embedder(3, num_freqs)
                pc_pos_dim = self.fourier_embedder_pc.out_dim
                self.fourier_embedder_img = get_embedder(6, num_freqs)
                img_pos_dim = self.fourier_embedder_img.out_dim
            else:
                pc_pos_dim = 3
                img_pos_dim = 6

            self.pc_pos_embed = nn.Sequential(
                nn.Linear(pc_pos_dim, self.dim_pc),
                nn.ReLU(),
                zero_module(nn.Linear(self.dim_pc, self.dim_pc))
            )

            self.img_pos_embed = nn.Sequential(
                nn.Linear(img_pos_dim, self.dim_img),
                nn.ReLU(),
                zero_module(nn.Linear(self.dim_img, self.dim_img))
            )

        depth_embed = None 
        self.depth_embed = depth_embed
        if depth_embed:
            if depth_embed == "fourier":
                self.fourier_embedder_depth = get_embedder(1, num_freqs)
                depth_in_dim = self.fourier_embedder_depth.out_dim
            else:
                depth_in_dim = 1
            self.pc_depth_pos_embed = nn.Sequential(
                nn.Linear(depth_in_dim, self.dim_img),
                nn.ReLU(),
                nn.Linear(self.dim_img, self.dim_img)
            )
            self.img_depth_pos_embed = nn.Sequential(
                nn.Linear(depth_in_dim, self.dim_pc),
                nn.ReLU(),
                nn.Linear(self.dim_pc, self.dim_pc)
            )

        # multiview attention
        self.norm1 = (
            AdaLayerNorm(dim_pc, num_embeds_ada_norm)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim_pc, elementwise_affine=norm_elementwise_affine)
        )
        self.norm2 = (
            AdaLayerNorm(dim_img, num_embeds_ada_norm)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim_img, elementwise_affine=norm_elementwise_affine)
        )
        self.attn1 = Attention(
            query_dim=dim_pc,
            cross_attention_dim=dim_img,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        self.attn2 = Attention(
            query_dim=dim_img,
            cross_attention_dim=dim_pc,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        if zero_module_type == "zero_linear":
            # NOTE: zero_module cannot apply to successive layers.
            self.connector1 = zero_module(nn.Linear(dim_pc, dim_pc))
            self.connector2 = zero_module(nn.Linear(dim_img, dim_img))

        elif zero_module_type == "gated":
            self.connector1 = GatedConnector(dim_pc)
            self.connector2 = GatedConnector(dim_img)

        elif zero_module_type == "none":
            # TODO: if this block is in controlnet, we may not need zero here.
            self.connector1 = lambda x: x
            self.connector2 = lambda x: x

        else:
            raise TypeError(f"Unknown zero module type: {zero_module_type}")

    @property
    def new_module(self):
        ret = {
            "norm1": self.norm1,
            "norm2": self.norm2,
            "attn1": self.attn1,
            "attn2": self.attn2,
        }
        if isinstance(self.connector1, nn.Module):
            ret["connector1"] = self.connector1
        if isinstance(self.connector2, nn.Module):
            ret["connector2"] = self.connector2
        if self.query_pos_embed:
            ret["pc_pos"] = self.pc_pos_embed
            ret["img_pos"] = self.img_pos_embed
        if self.depth_embed:
            ret["pc_depth"] = self.pc_depth_pos_embed
            ret["img_depth"] = self.img_depth_pos_embed           
        return ret

    def _img_sample_pts_along_rays(self, lidar2imgs, height, width):
        # sample_depths = torch.linspace(1, 60, self.img_ray_sample_num).type_as(lidar2imgs)
        index = torch.arange(start=0, end=self.img_ray_sample_num, step=1, device=lidar2imgs.device).float()
        index_1 = index + 1
        bin_size = (self.img_max_depth - 1) / (self.img_ray_sample_num * (1 + self.img_ray_sample_num))
        sample_depths = 1 + bin_size * index * index_1

        xs = np.arange(width) + 0.5
        ys = np.arange(height) + 0.5
        
        xs = xs * (self.image_size[0] / width)
        ys = ys * (self.image_size[1] / height)

        xs, ys = np.meshgrid(xs, ys, indexing='xy')

        coords_img_2d = np.stack([xs, ys], axis=-1)  # [height, width, 2]
        coords_img_2d = torch.from_numpy(coords_img_2d).type_as(sample_depths)  # [height, width, 2]
        coords_img_2d = coords_img_2d[:, :, None].repeat(1, 1, self.img_ray_sample_num, 1)  # [height, width, sample_num, 2]
        sample_depths = sample_depths[None, None].repeat(height, width, 1)
        coords_img_3d = torch.cat([coords_img_2d, sample_depths[..., None]], dim=-1)  # [height, width, sample_num, 3]

        coords_img_3d[..., :2] = coords_img_3d[..., :2] * coords_img_3d[..., 2:3]
        coords_img_4d = torch.cat([coords_img_3d, torch.ones_like(coords_img_3d[..., :1])], dim=-1)  # [height, width, sample_num, 4]

        coords_img_4d = coords_img_4d[None, None, :, :, :, :, None]  # [1, 1, height, width, sample_num, 4, 1]
        coords_4d = torch.matmul(torch.inverse(lidar2imgs[:, :, None, None, None].float()).type_as(coords_img_4d), coords_img_4d).squeeze(-1)  # [bs, num_views, height, width, sample_num, 4]

        coords_3d = coords_4d[..., :3]  # [bs, num_views, height, width, sample_num, 3]

        return coords_3d


    def _pc_sample_pts_along_rays(self, lidar2imgs, height, width):
        device = lidar2imgs.device
        # sample_depths = torch.linspace(1, 60, self.pc_ray_sample_num).to(device)
        index = torch.arange(start=0, end=self.pc_ray_sample_num, step=1, device=device).float()
        index_1 = index + 1
        bin_size = (self.pc_max_range - 1) / (self.pc_ray_sample_num * (1 + self.pc_ray_sample_num))
        sample_depths = 1 + bin_size * index * index_1

        yaw = (torch.arange(width) + 0.5) / width * 2 * math.pi
        pitch = self.fov[1] - (torch.arange(height) + 0.5) / height * (self.fov[1] - self.fov[0])

        yaw = yaw.to(device)
        pitch = pitch.to(device)

        zs = torch.sin(pitch[:, None]).repeat(1, yaw.shape[0])
        ys = torch.cos(pitch[:, None]) * torch.sin(yaw[None])
        xs = torch.cos(pitch[:, None]) * torch.cos(yaw[None])

        coords_3d = torch.stack([xs, ys, zs], dim=-1)
        coords_3d = coords_3d[:, :, None] * sample_depths[None, None, :, None]

        return coords_3d

    def _epipolar_projection_pc(self, sample_pc, sample_img, lidar2imgs):
        pc_height, pc_width = sample_pc.shape[1:3]
        img_height, img_width = sample_img.shape[2:4]
        bs, n_cam = sample_img.shape[:2]

        coords_proj = torch.rand([bs, n_cam, pc_height, pc_width, self.pc_ray_sample_num, 4]).type_as(sample_pc)
        proj_mask = (coords_proj[..., 0]>0) & (coords_proj[..., 0]<img_width) & (coords_proj[..., 1]>0) & (coords_proj[..., 1]<img_height) & (coords_proj[..., 2]>0)


        coords_proj = coords_proj[..., :2]

        coords_proj = coords_proj.flatten(2,3).flatten(0,1)
        coords_proj[..., 0] = coords_proj[..., 0] / img_width * 2 - 1
        coords_proj[..., 1] = coords_proj[..., 1] / img_height * 2 - 1

        proj_feat = F.grid_sample(sample_img.flatten(0, 1).permute(0, 3, 1, 2), coords_proj, align_corners=False)

        proj_feat = proj_feat.view(bs, n_cam, self.dim_img, pc_height, pc_width, self.pc_ray_sample_num)

        proj_feat = proj_feat * proj_mask[:, :, None]
        proj_feat = torch.sum(proj_feat, dim=1) / (torch.sum(proj_mask[:, :, None], dim=1) + 1e-5)

        proj_feat = proj_feat.permute(0, 2, 3, 4, 1)
        proj_mask = torch.sum(proj_mask, dim=1) > 0

        return proj_feat, proj_mask


    def _epipolar_projection_img(self, sample_img, sample_pc, lidar2imgs):
        pc_height, pc_width = sample_pc.shape[1:3]
        img_height, img_width = sample_img.shape[2:4]
        bs, n_cam = sample_img.shape[:2]

        yaw = torch.rand([bs, n_cam, img_height, img_width, self.img_ray_sample_num]).type_as(sample_img) * 2 - 1
        pitch = torch.rand([bs, n_cam, img_height, img_width, self.img_ray_sample_num]).type_as(sample_img) * 2 - 1
        coords_pc = torch.stack([yaw, pitch], dim=-1)
        proj_mask = (yaw > -1) & (yaw < 1) & (pitch > -1) & (pitch < 1)
        
        coords_pc = coords_pc.flatten(1, 3)
        proj_feat = F.grid_sample(sample_pc.permute(0, 3, 1, 2), coords_pc, align_corners=False)
        proj_feat = proj_feat.view(bs, self.dim_pc, n_cam, img_height, img_width, self.img_ray_sample_num)

        proj_feat = proj_feat * proj_mask[:, None]
        proj_feat = proj_feat.permute(0, 2, 3, 4, 5, 1)

        return proj_feat, proj_mask

    def forward(
        self,
        sample_pc,
        sample_img,
        lidar2imgs,
        timestep=None,
    ):

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        bs = sample_pc.shape[0]
        n_cam = lidar2imgs.shape[1]
        pc_h, pc_w = sample_pc.shape[-2:]
        img_h, img_w = sample_img.shape[-2:]

        sample_pc = sample_pc.permute(0, 2, 3, 1)
        sample_img = sample_img.permute(0, 2, 3, 1)
        sample_pc = sample_pc.view(bs, pc_h, pc_w, self.dim_pc)
        sample_img = sample_img.view(bs, n_cam, img_h, img_w, self.dim_img)
            

        sample_pc_ = sample_pc.clone()
        sample_img_ = sample_img.clone()

        if self.use_ada_layer_norm:
            sample_pc = self.norm1(sample_pc, timestep)
            sample_img = self.norm2(sample_img, timestep)

        else:
            sample_pc = self.norm1(sample_pc)
            sample_img = self.norm2(sample_img)

        pc_proj_features, pc_proj_mask = self._epipolar_projection_pc(sample_pc, sample_img, lidar2imgs)
        img_proj_features, img_proj_mask = self._epipolar_projection_img(sample_img, sample_pc, lidar2imgs)

        pc_proj_mask = pc_proj_mask.float()
        pc_proj_mask = torch.where(pc_proj_mask>0, 0.0, -1e4)

        img_proj_mask = img_proj_mask.float()
        img_proj_mask = torch.where(img_proj_mask>0, 0.0, -1e4)

        attn_output = self.attn1(sample_pc.flatten(0,2)[:, None], encoder_hidden_states=pc_proj_features.flatten(0, 2), attention_mask=pc_proj_mask.flatten(0, 2)[:, None])
        attn_output = attn_output.view(bs, pc_h, pc_w, self.dim_pc)

        attn_output = self.connector1(attn_output)

        sample_pc = sample_pc_ + attn_output

        if self.depth_embed:
            index = torch.arange(start=0, end=self.img_ray_sample_num, step=1, device=lidar2imgs.device).float()
            index_1 = index + 1
            bin_size = (self.img_max_depth - 1) / (self.img_ray_sample_num * (1 + self.img_ray_sample_num))
            img_depth = 1 + bin_size * index * index_1
            img_depth = img_depth[..., None]
            if self.depth_embed == 'fourier':
                img_depth_embed = self.fourier_embedder_depth(img_depth / self.img_max_depth)
            else:
                img_depth_embed = img_depth
            img_depth_embed = self.img_depth_pos_embed(img_depth_embed)
            img_proj_features = img_proj_features + img_depth_embed[None, None, None, None]

        attn_output = self.attn2(sample_img.flatten(0,3)[:, None], encoder_hidden_states=img_proj_features.flatten(0, 3), attention_mask=img_proj_mask.flatten(0, 3)[:, None])
        attn_output = attn_output.view(bs, n_cam, img_h, img_w, self.dim_img)
        attn_output = self.connector2(attn_output)
        sample_img = sample_img_ + attn_output

        sample_pc = sample_pc.permute(0, 3, 1, 2)
        sample_img = sample_img.permute(0, 1, 4, 2, 3).flatten(0, 1)

        return sample_pc, sample_img

