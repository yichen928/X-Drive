import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import copy

from diffusers import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.configuration_utils import register_to_config
from diffusers.models.unet_2d_condition import (
    UNet2DConditionModel,
    UNet2DConditionOutput,
)
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    UNetMidBlock2DSimpleCrossAttn,
    UpBlock2D,

)
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor
from diffusers.models.activations import get_activation
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders import UNet2DConditionLoadersMixin

from ..misc.common import _get_module, _set_module

from diffusers.models.resnet import Downsample2D

from .blocks_multi import MultiModalityTransformerBlock, RandomMultiModalityTransformerBlock

class UNetMultiRangeLDMBoxConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        img_unet,
        pc_unet,
        down_pc_block_channels,
        down_img_block_channels,
        down_cross_attn_head_dims,
        down_img_scale_factors,

        up_pc_block_channels,
        up_img_block_channels,
        up_cross_attn_head_dims,
        up_img_scale_factors,

        zero_module_type,

        pc_ray_sample_num,
        img_ray_sample_num,
        query_pos_embed='3d-mlp',
        fov=(-30, 10),

        depth_embed=None,
        pc_max_range=80,
        img_max_depth=80,
    ):

        super().__init__()

        self.img_unet = img_unet
        self.pc_unet = pc_unet

        self.zero_module_type = zero_module_type

        if isinstance(pc_ray_sample_num, int):
            pc_ray_sample_num = [pc_ray_sample_num] * (len(down_pc_block_channels) + len(up_pc_block_channels))

        if isinstance(img_ray_sample_num, int):
            img_ray_sample_num = [img_ray_sample_num] * (len(down_pc_block_channels) + len(up_pc_block_channels))
        multiblock_params = {
            "query_pos_embed": query_pos_embed,
            "fov": fov,
            # "pc_ray_sample_num": pc_ray_sample_num,
            # "img_ray_sample_num": img_ray_sample_num,
            "depth_embed": depth_embed,
            "pc_max_range": pc_max_range,
            "img_max_depth": img_max_depth,
        }

        assert len(down_pc_block_channels) == len(down_img_block_channels)

        down_blocks = []
        for i in range(len(down_pc_block_channels)):
            down_block = MultiModalityTransformerBlock(
                dim_pc=down_pc_block_channels[i], dim_img=down_img_block_channels[i], num_attention_heads=8, attention_head_dim=down_cross_attn_head_dims[i], img_scale_factor=down_img_scale_factors[i],
                pc_ray_sample_num = pc_ray_sample_num[i], img_ray_sample_num = img_ray_sample_num[i],
                **multiblock_params,
            )
            # down_block = RandomMultiModalityTransformerBlock(
            #     dim_pc=down_pc_block_channels[i], dim_img=down_img_block_channels[i], num_attention_heads=8, attention_head_dim=down_cross_attn_head_dims[i], img_scale_factor=down_img_scale_factors[i],
            #     pc_ray_sample_num = pc_ray_sample_num[i], img_ray_sample_num = img_ray_sample_num[i],
            #     **multiblock_params,
            # )
            down_blocks.append(down_block)
        self.multi_down_blocks = nn.ModuleList(down_blocks)

        assert len(up_pc_block_channels) == len(up_img_block_channels)
        up_blocks = []
        for i in range(len(up_pc_block_channels)):
            up_block = MultiModalityTransformerBlock(
                dim_pc=up_pc_block_channels[i], dim_img=up_img_block_channels[i], num_attention_heads=8, attention_head_dim=up_cross_attn_head_dims[i], img_scale_factor=up_img_scale_factors[i],
                pc_ray_sample_num = pc_ray_sample_num[i+len(down_pc_block_channels)], img_ray_sample_num = img_ray_sample_num[i+len(down_pc_block_channels)],
                **multiblock_params,
            )
            # up_block = RandomMultiModalityTransformerBlock(
            #     dim_pc=up_pc_block_channels[i], dim_img=up_img_block_channels[i], num_attention_heads=8, attention_head_dim=up_cross_attn_head_dims[i], img_scale_factor=up_img_scale_factors[i],
            #     pc_ray_sample_num = pc_ray_sample_num[i+len(down_pc_block_channels)], img_ray_sample_num = img_ray_sample_num[i+len(down_pc_block_channels)],
            #     **multiblock_params,
            # )
            up_blocks.append(up_block)
        self.multi_up_blocks = nn.ModuleList(up_blocks)

    @property
    def trainable_module(self) -> Dict[str, nn.Module]:
        ret_dict = {}
        for key, value in self.pc_unet.trainable_module.items():
            ret_dict["pc_"+key] = value
        for key, value in self.img_unet.trainable_module.items():
            ret_dict["img_"+key] = value          
        for i, mod in enumerate(self.multi_down_blocks):
            for key, value in mod.new_module.items():
                ret_dict["down_"+str(i)+"_"+key] = value 
        for i, mod in enumerate(self.multi_up_blocks):
            for key, value in mod.new_module.items():
                ret_dict["up_"+str(i)+"_"+key] = value
        return ret_dict

    @property
    def trainable_parameters(self) -> List[nn.Parameter]:
        params = []
        for mod in self.trainable_module.values():
            for param in mod.parameters():
                params.append(param)
        return params



    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        # first, set all to false
        super().train(False)
        if mode:
            # ensure gradient_checkpointing is usable, set training = True
            for mod in self.modules():
                if getattr(mod, "gradient_checkpointing", False):
                    mod.training = True
        # then, for some modules, we set according to `mode`
        self.training = False
        for mod in self.trainable_module.values():
            if mod is self:
                super().train(mode)
            else:
                mod.train(mode)
        return self

    # def enable_gradient_checkpointing(self, flag=None):
    #     """
    #     Activates gradient checkpointing for the current model.

    #     Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
    #     activations".
    #     """
    #     # self.apply(partial(self._set_gradient_checkpointing, value=True))
    #     mod_idx = -1
    #     for module in self.modules():
    #         if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D, CrossAttnUpBlock2D, UpBlock2D)):
    #             mod_idx += 1
    #             if flag is not None and not flag[mod_idx]:
    #                 logging.debug(
    #                     f"[UNet2DConditionModelMultiview] "
    #                     f"gradient_checkpointing skip [{module.__class__}]")
    #                 continue
    #             logging.debug(f"[UNet2DConditionModelMultiview] set "
    #                           f"[{module.__class__}] to gradient_checkpointing")
    #             module.gradient_checkpointing = True

    def downsample_step(self, downsample_block, sample, emb, encoder_hidden_states, attention_mask=None):
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
            )
        else:
            sample, res_samples = downsample_block(
                hidden_states=sample, temb=emb,
            )

        return sample, res_samples
    
    def upsample_step(self, upsample_block, sample, emb, res_samples, encoder_hidden_states, upsample_size, attention_mask=None):
        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            sample = upsample_block(
                hidden_states=sample, temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                upsample_size=upsample_size,
                encoder_attention_mask=attention_mask,
            )
        else:
            sample = upsample_block(
                hidden_states=sample, temb=emb,
                res_hidden_states_tuple=res_samples,
                upsample_size=upsample_size,
            )
        
        return sample


    def forward(
        self,
        sample_pc: torch.FloatTensor,
        sample_img: torch.FloatTensor,
        timesteps_pc: Union[torch.Tensor, int, float],
        timesteps_img: Union[torch.Tensor, int, float],
        encoder_hidden_states_pc: torch.Tensor,
        encoder_hidden_states_img: torch.Tensor,
        lidar2imgs: torch.Tensor,
        attention_mask_pc: Optional[torch.Tensor] = None,
        attention_mask_img: Optional[torch.Tensor] = None,

        timestep_cond = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # TODO: actually, we do not change logic in forward

        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size_img = True
        forward_upsample_size_pc = False

        device = sample_pc.device

        # prepare attention_mask
        if attention_mask_pc is not None:
            attention_mask_pc = (1 - attention_mask_pc.to(sample_pc.dtype)) * -10000.0
            attention_mask_pc = attention_mask_pc.unsqueeze(1)

        if attention_mask_img is not None:
            attention_mask_img = (1 - attention_mask_img.to(sample_img.dtype)) * -10000.0
            attention_mask_img = attention_mask_img.unsqueeze(1)

        if len(encoder_hidden_states_pc) == 3:
            encoder_hidden_states_pc[-1] = (1 - encoder_hidden_states_pc[-1].to(sample_pc.dtype)) * -10000.0
            encoder_hidden_states_pc[-1] = encoder_hidden_states_pc[-1].unsqueeze(1)

        if len(encoder_hidden_states_img) == 3:
            encoder_hidden_states_img[-1] = (1 - encoder_hidden_states_img[-1].to(sample_img.dtype)) * -10000.0
            encoder_hidden_states_img[-1] = encoder_hidden_states_img[-1].unsqueeze(1)

        # 0. center input if necessary
        if self.pc_unet.config.center_input_sample:
            sample_pc = 2 * sample_pc - 1.0
        if self.img_unet.config.center_input_sample:
            sample_img = 2 * sample_img - 1.0

        # 1. time
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML

        if not torch.is_tensor(timesteps_pc):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = device.type == "mps"
            if isinstance(timesteps_pc, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps_pc = torch.tensor(
                [timesteps_pc],
                dtype=dtype, device=device)
        elif len(timesteps_pc.shape) == 0:
            timesteps_pc = timesteps_pc[None].to(device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps_pc = timesteps_pc.expand(sample_pc.shape[0])

        if not torch.is_tensor(timesteps_img):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = device.type == "mps"
            if isinstance(timesteps_img, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps_img = torch.tensor(
                [timesteps_img],
                dtype=dtype, device=device)
        elif len(timesteps_img.shape) == 0:
            timesteps_img = timesteps_img[None].to(device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps_img = timesteps_img.expand(timesteps_img.shape[0])


        t_emb_pc = self.pc_unet.time_proj(timesteps_pc)
        t_emb_img = self.img_unet.time_proj(timesteps_img)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb_pc = t_emb_pc.to(dtype=self.dtype)
        t_emb_img = t_emb_img.to(dtype=self.dtype)

        emb_pc = self.pc_unet.time_embedding(t_emb_pc, timestep_cond)
        emb_img = self.img_unet.time_embedding(t_emb_img, timestep_cond)

        # if self.pc_unet.time_embed_act is not None:
        #     emb_pc = self.pc_unet.time_embed_act(emb_pc)
        if self.img_unet.time_embed_act is not None:
            emb_img = self.img_unet.time_embed_act(emb_img)

        # if self.pc_unet.encoder_hid_proj is not None:
        #     encoder_hidden_states_pc = self.pc_unet.encoder_hid_proj(encoder_hidden_states_pc)

        if self.img_unet.encoder_hid_proj is not None:
            encoder_hidden_states_img = self.img_unet.encoder_hid_proj(encoder_hidden_states_img)

        # 2. pre-process
        sample_pc = self.pc_unet.conv_in(sample_pc)
        sample_img = self.img_unet.conv_in(sample_img)

        # 3. down
        down_block_res_samples_pc = (sample_pc,)
        down_block_res_samples_img = (sample_img,)

        num_down_blocks = max(len(self.pc_unet.down_blocks), len(self.img_unet.down_blocks))
        for block_id in range(num_down_blocks):
            if block_id != num_down_blocks - 1:
                sample_pc = sample_pc.permute(0, 1, 3, 2)
                sample_pc, sample_img = self.multi_down_blocks[block_id](sample_pc, sample_img, lidar2imgs)
                sample_pc = sample_pc.permute(0, 1, 3, 2)
            if block_id < len(self.img_unet.down_blocks):
                downsample_block_img = self.img_unet.down_blocks[block_id]
                sample_img, res_samples_img = self.downsample_step(downsample_block_img, sample_img, emb_img, encoder_hidden_states_img, attention_mask=attention_mask_img)
                down_block_res_samples_img += res_samples_img
            if block_id < len(self.pc_unet.down_blocks):
                downsample_block_pc = self.pc_unet.down_blocks[block_id]
                sample_pc, res_samples_pc = self.downsample_step(downsample_block_pc, sample_pc, emb_pc, encoder_hidden_states_pc, attention_mask=attention_mask_pc)
                down_block_res_samples_pc += res_samples_pc

        # 4. mid
        sample_pc = sample_pc.permute(0, 1, 3, 2)
        sample_pc, sample_img = self.multi_down_blocks[-1](sample_pc, sample_img, lidar2imgs)
        sample_pc = sample_pc.permute(0, 1, 3, 2)
        if self.pc_unet.mid_block is not None:
            sample_pc = self.pc_unet.mid_block(
                sample_pc,
                temb=emb_pc,
                encoder_hidden_states=encoder_hidden_states_pc,
                encoder_attention_mask=attention_mask_pc,
            )
        if self.img_unet.mid_block is not None:
            sample_img = self.img_unet.mid_block(
                sample_img,
                emb_img,
                encoder_hidden_states=encoder_hidden_states_img,
                encoder_attention_mask=attention_mask_img,
            )

        # down_block_res_samples = down_block_res_samples[:-1]
        # 5. up
        num_up_blocks = max(len(self.pc_unet.up_blocks), len(self.img_unet.up_blocks))
        pc_block_id = 0
        img_block_id = 0
        upsample_size_pc = None 
        upsample_size_img = None
        for i in range(num_up_blocks):
            if i != 0:
                sample_pc = sample_pc.permute(0, 1, 3, 2)
                sample_pc, sample_img = self.multi_up_blocks[i-1](sample_pc, sample_img, lidar2imgs)
                sample_pc = sample_pc.permute(0, 1, 3, 2)
            if i+len(self.pc_unet.up_blocks) >= num_up_blocks:
                upsample_block_pc = self.pc_unet.up_blocks[pc_block_id]
                is_final_block_pc = pc_block_id == len(self.pc_unet.up_blocks) - 1
                res_samples_pc = down_block_res_samples_pc[-len(upsample_block_pc.resnets):]
                down_block_res_samples_pc = down_block_res_samples_pc[: -len(upsample_block_pc.resnets)]
                if not is_final_block_pc and forward_upsample_size_pc:
                    upsample_size_pc = down_block_res_samples_pc[-1].shape[2:]

                sample_pc = self.upsample_step(upsample_block_pc, sample_pc, emb_pc, res_samples_pc, encoder_hidden_states_pc, upsample_size_pc, attention_mask=attention_mask_pc)
                pc_block_id += 1

                
            if i+len(self.img_unet.up_blocks) >= num_up_blocks:
                upsample_block_img = self.img_unet.up_blocks[img_block_id]
                is_final_block_img = img_block_id == len(self.img_unet.up_blocks) - 1
                res_samples_img = down_block_res_samples_img[-len(upsample_block_img.resnets):]
                down_block_res_samples_img = down_block_res_samples_img[: -len(upsample_block_img.resnets)]
                if not is_final_block_img and forward_upsample_size_img:
                    upsample_size_img = down_block_res_samples_img[-1].shape[2:]
                sample_img = self.upsample_step(upsample_block_img, sample_img, emb_img, res_samples_img, encoder_hidden_states_img, upsample_size_img, attention_mask=attention_mask_img)
                img_block_id += 1
                    
        # 6. post-process
        if self.pc_unet.conv_norm_out:
            sample_pc = self.pc_unet.conv_norm_out(sample_pc)
            sample_pc = self.pc_unet.conv_act(sample_pc)
        sample_pc = self.pc_unet.conv_out(sample_pc)

        if self.img_unet.conv_norm_out:
            sample_img = self.img_unet.conv_norm_out(sample_img)
            sample_img = self.img_unet.conv_act(sample_img)
        sample_img = self.img_unet.conv_out(sample_img)

        if not return_dict:
            return (sample_pc, sample_img)

        return UNet2DConditionOutput(sample=sample_pc), UNet2DConditionOutput(sample=sample_img)

