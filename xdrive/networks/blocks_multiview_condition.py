from typing import Any, Dict, List, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

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


class BasicMultiViewConditionTransformerBlock(BasicTransformerBlock):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
        # multi_view
        neighboring_view_pair: Optional[Dict[int, List[int]]] = None,
        neighboring_attn_type: Optional[str] = "add",
        zero_module_type="zero_linear",
        cross_view_type="default",
        img_pos_embedding=False,
        num_freqs=4,
        img_size=None,
        separate_condition=True,
        use_gsa=False,
    ):
        super().__init__(
            dim, num_attention_heads, attention_head_dim, dropout,
            cross_attention_dim, activation_fn, num_embeds_ada_norm,
            attention_bias, only_cross_attention, double_self_attention,
            upcast_attention, norm_elementwise_affine, norm_type, final_dropout)
        self.use_gsa = use_gsa
        if use_gsa:
            separate_condition = True
        self.neighboring_view_pair = _ensure_kv_is_int(neighboring_view_pair)
        self.neighboring_attn_type = neighboring_attn_type
        self.cross_view_type = cross_view_type
        assert cross_view_type in ["default", "half", "none"]
        self.img_size = img_size

        self.separate_condition = separate_condition

        # multiview attention
        if self.separate_condition:
            self.norm4 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
            if self.use_gsa:
                self.attn4_proj = nn.Sequential(
                    nn.Linear(cross_attention_dim, dim),
                    nn.LayerNorm(dim)
                )
                self.attn4 = Attention(
                    query_dim=dim,
                    cross_attention_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                )
            else:
                self.attn4 = Attention(
                    query_dim=dim,
                    cross_attention_dim=cross_attention_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                )

        if self.cross_view_type != "none":
            # multiview attention
            self.norm5 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
            self.attn5 = Attention(
                query_dim=dim,
                cross_attention_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        
        if zero_module_type == "zero_linear":
            # NOTE: zero_module cannot apply to successive layers.
            if self.separate_condition:
                self.connector4 = zero_module(nn.Linear(dim, dim))
            if self.cross_view_type != "none":
                self.connector5 = zero_module(nn.Linear(dim, dim))
        elif zero_module_type == "gated":
            if self.separate_condition:
                self.connector4 = GatedConnector(dim)
            if self.cross_view_type != "none":
                self.connector5 = GatedConnector(dim)
        elif zero_module_type == "none":
            # TODO: if this block is in controlnet, we may not need zero here.
            if self.separate_condition:
                self.connector4 = lambda x: x
            if self.cross_view_type != "none":
                self.connector5 = lambda x: x
        else:
            raise TypeError(f"Unknown zero module type: {zero_module_type}")
        
        self.img_pos_embedding = img_pos_embedding
        if img_pos_embedding:
            self.embedder = get_embedder(2, num_freqs)
            self.pos_encoding = nn.Sequential(
                nn.Linear(self.embedder.out_dim, dim),
                nn.SiLU(),
                zero_module(nn.Linear(dim, dim))
            )

    @property
    def new_module(self):
        if self.cross_view_type != "none":
            ret = {
                "norm5": self.norm5,
                "attn5": self.attn5,
            }
            if isinstance(self.connector5, nn.Module):
                ret["connector5"] = self.connector5
        else:
            ret = {}
        if self.separate_condition:
            ret["norm4"] = self.norm4
            ret["attn4"] = self.attn4
            if self.use_gsa:
                ret["attn4_proj"] = self.attn4_proj
        else:
            ret["norm2"] = self.norm2
            ret["attn2"] = self.attn2       

        if self.separate_condition and isinstance(self.connector4, nn.Module):
            ret["connector4"] = self.connector4
        if self.img_pos_embedding:
            ret["img_pos_encoding"] = self.pos_encoding

        return ret

    @property
    def n_cam(self):
        return len(self.neighboring_view_pair)


    def _construct_attn_input_half(self, norm_hidden_states, height, width):
        B = len(norm_hidden_states)
        num_views = norm_hidden_states.shape[1]
        dim = norm_hidden_states.shape[3]
        norm_hidden_states = norm_hidden_states.view(B, num_views, height, width, dim)

        # reshape, key for origin view, value for ref view
        hidden_states_in1 = []
        hidden_states_in2 = []
        cam_order = []
        for key, values in self.neighboring_view_pair.items():
            assert len(values) == 2
            hidden_states_in1.append(norm_hidden_states[:, key, :, :math.ceil(width/2)])
            hidden_states_in2.append(norm_hidden_states[:, values[0], :, -math.ceil(width/2):])

            hidden_states_in1.append(norm_hidden_states[:, key, :, -math.ceil(width/2):])
            hidden_states_in2.append(norm_hidden_states[:, values[1], :, :math.ceil(width/2)])
            cam_order += [key] * B * 2
        # N*B, H*W, head*dim
        hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
        # N*B, 2*H*W, head*dim
        hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
        cam_order = torch.LongTensor(cam_order)

        hidden_states_in1 = hidden_states_in1.flatten(1, 2)
        hidden_states_in2 = hidden_states_in2.flatten(1, 2)

        return hidden_states_in1, hidden_states_in2, cam_order


    def _construct_attn_input(self, norm_hidden_states):
        B = len(norm_hidden_states)
        # reshape, key for origin view, value for ref view
        hidden_states_in1 = []
        hidden_states_in2 = []
        cam_order = []
        if self.neighboring_attn_type == "add":
            for key, values in self.neighboring_view_pair.items():
                for value in values:
                    hidden_states_in1.append(norm_hidden_states[:, key])
                    hidden_states_in2.append(norm_hidden_states[:, value])
                    cam_order += [key] * B
            # N*2*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
        elif self.neighboring_attn_type == "concat":
            for key, values in self.neighboring_view_pair.items():
                hidden_states_in1.append(norm_hidden_states[:, key])
                hidden_states_in2.append(torch.cat([
                    norm_hidden_states[:, value] for value in values
                ], dim=1))
                cam_order += [key] * B
            # N*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            # N*B, 2*H*W, head*dim
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
        elif self.neighboring_attn_type == "self":
            hidden_states_in1 = rearrange(
                norm_hidden_states, "b n l ... -> b (n l) ...")
            hidden_states_in2 = None
            cam_order = None
        else:
            raise NotImplementedError(
                f"Unknown type: {self.neighboring_attn_type}")
        return hidden_states_in1, hidden_states_in2, cam_order

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ):
        encoder_hidden_states_text = encoder_hidden_states[0]
        encoder_hidden_states_bbox = encoder_hidden_states[1]
        # bbox_mask = encoder_hidden_states[2].type_as(encoder_hidden_states_bbox)
        # bbox_mask = torch.where(bbox_mask>0, 0.0, -1e4)
        if len(encoder_hidden_states) == 3:
            bbox_mask = encoder_hidden_states[2]
        else:
            bbox_mask = None

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        attn_output = self.attn1(
            norm_hidden_states, encoder_hidden_states=encoder_hidden_states_text
            if self.only_cross_attention else None,
            attention_mask=attention_mask, **cross_attention_kwargs,)
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm else self.norm2(hidden_states))
            # TODO (Birch-San): Here we should prepare the encoder_attention mask correctly
            # prepare attention mask here

        if not self.separate_condition:
            condition_hidden_states = torch.cat([encoder_hidden_states_text, encoder_hidden_states_bbox], dim=1) 
            condition_mask = torch.cat([encoder_attention_mask, bbox_mask], dim=2)
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=condition_hidden_states,
                attention_mask=condition_mask,
                **cross_attention_kwargs,
            )
        else:
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states_text,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
        hidden_states = attn_output + hidden_states


        scale = round(math.log(math.sqrt(self.img_size[0] * self.img_size[1] // hidden_states.shape[1]), 2))
        scale = 2 ** scale
        height = math.ceil(self.img_size[0] / scale)
        width = hidden_states.shape[1] // height

        # bbox conditional cross attention
        norm_hidden_states = hidden_states
        if self.img_pos_embedding:
            x_pos = torch.arange(width).type_as(norm_hidden_states) + 0.5
            y_pos = torch.arange(height).type_as(norm_hidden_states) + 0.5
            x_pos = x_pos / width
            y_pos = y_pos / height
            xy_pos = torch.meshgrid(x_pos, y_pos, indexing='xy')
            xy_pos = torch.stack(xy_pos, dim=-1)

            xy_pos = self.embedder(xy_pos)

            xy_pos = self.pos_encoding(xy_pos)

            xy_pos = xy_pos.flatten(0,1)[None]
            norm_hidden_states = norm_hidden_states + xy_pos

        if self.separate_condition:
            norm_hidden_states = (
                self.norm4(norm_hidden_states, timestep) if self.use_ada_layer_norm else
                self.norm4(norm_hidden_states)
            )

            # attention
            if self.use_gsa:
                bbox_input = self.attn4_proj(encoder_hidden_states_bbox)
                bbox_hidden_mask = torch.cat([bbox_mask, torch.zeros_like(norm_hidden_states[:, None, :, 0])], dim=-1)
                attn_output = self.attn4(
                    norm_hidden_states,
                    encoder_hidden_states=torch.cat([bbox_input, norm_hidden_states], dim=1),
                    attention_mask=bbox_hidden_mask,
                    **cross_attention_kwargs,
                )
            else:
                attn_output = self.attn4(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states_bbox,
                    attention_mask=bbox_mask
                )

            # apply zero init connector (one layer)
            attn_output = self.connector4(attn_output)
            # short-cut
            hidden_states = attn_output + hidden_states

        if self.cross_view_type != "none":
            # multi-view cross attention
            norm_hidden_states = (
                self.norm5(hidden_states, timestep) if self.use_ada_layer_norm else
                self.norm5(hidden_states)
            )
            # batch dim first, cam dim second
            norm_hidden_states = rearrange(
                norm_hidden_states, '(b n) ... -> b n ...', n=self.n_cam)
            B = len(norm_hidden_states)

            if self.cross_view_type == "default":
                # key is query in attention; value is key-value in attention
                hidden_states_in1, hidden_states_in2, cam_order = self._construct_attn_input(
                    norm_hidden_states, )
            else:
                hidden_states_in1, hidden_states_in2, cam_order = self._construct_attn_input_half(
                    norm_hidden_states, height=height, width=width)    

            # attention
            attn_raw_output = self.attn5(
                hidden_states_in1,
                encoder_hidden_states=hidden_states_in2,
                **cross_attention_kwargs,
            )
            # final output
            if self.neighboring_attn_type == "self":
                attn_output = rearrange(
                    attn_raw_output, 'b (n l) ... -> b n l ...', n=self.n_cam)
            else:
                attn_output = torch.zeros_like(norm_hidden_states)
                if self.cross_view_type == "default":
                    for cam_i in range(self.n_cam):
                        attn_out_mv = rearrange(attn_raw_output[cam_order == cam_i],
                                                '(n b) ... -> b n ...', b=B)
                        attn_output[:, cam_i] = torch.sum(attn_out_mv, dim=1)
                else:
                    # scale = round(math.log(math.sqrt(self.img_size[0] * self.img_size[1] // norm_hidden_states.shape[2]), 2))
                    # scale = 2 ** scale
                    # height = math.ceil(self.img_size[0] / scale)
                    # width = norm_hidden_states.shape[2] // height

                    for cam_i in range(self.n_cam):
                        attn_out_mv = attn_raw_output[cam_order == cam_i].view(2, B, height, math.ceil(width/2), -1)
                        attn_out_mv = torch.cat([attn_out_mv[0], attn_out_mv[1, :, :, -(width-math.ceil(width/2)):]], dim=2)
                        attn_output[:, cam_i] = attn_out_mv.flatten(1, 2)

            attn_output = rearrange(attn_output, 'b n ... -> (b n) ...')
            # apply zero init connector (one layer)
            attn_output = self.connector5(attn_output)
            # short-cut
            hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (
                1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states

