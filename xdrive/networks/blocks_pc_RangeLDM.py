import torch 
import math
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.controlnet import zero_module
from diffusers.models.unet_2d_blocks import *
from .embedder import PositionEmbedderFourierMLP3D, get_embedder


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    resnet_skip_time_act=False,
    resnet_out_scale_factor=1.0,
    cross_attention_norm=None,
    use_gsa=False,
):
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "ResnetDownsampleBlock2D":
        return ResnetDownsampleBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            skip_time_act=resnet_skip_time_act,
            output_scale_factor=resnet_out_scale_factor,
        )
    elif down_block_type == "AttnDownBlock2D":
        return AttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            attn_num_head_channels=attn_num_head_channels,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "CondAttnDownBlock2D":
        return CondAttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            resnet_time_scale_shift=resnet_time_scale_shift,
            use_gsa=use_gsa,
        )
    elif down_block_type == "CrossAttnDownBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock2D")
        return CrossAttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "SimpleCrossAttnDownBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for SimpleCrossAttnDownBlock2D")
        return SimpleCrossAttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            resnet_time_scale_shift=resnet_time_scale_shift,
            skip_time_act=resnet_skip_time_act,
            output_scale_factor=resnet_out_scale_factor,
            only_cross_attention=only_cross_attention,
            cross_attention_norm=cross_attention_norm,
        )
    elif down_block_type == "SkipDownBlock2D":
        return SkipDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "AttnSkipDownBlock2D":
        return AttnSkipDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            downsample_padding=downsample_padding,
            attn_num_head_channels=attn_num_head_channels,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "DownEncoderBlock2D":
        return DownEncoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "AttnDownEncoderBlock2D":
        return AttnDownEncoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            attn_num_head_channels=attn_num_head_channels,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "KDownBlock2D":
        return KDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
        )
    elif down_block_type == "KCrossAttnDownBlock2D":
        return KCrossAttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            add_self_attention=True if not add_downsample else False,
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    resnet_skip_time_act=False,
    resnet_out_scale_factor=1.0,
    cross_attention_norm=None,
    use_gsa=False,
):
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "ResnetUpsampleBlock2D":
        return ResnetUpsampleBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            skip_time_act=resnet_skip_time_act,
            output_scale_factor=resnet_out_scale_factor,
        )
    elif up_block_type == "CrossAttnUpBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock2D")
        return CrossAttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "SimpleCrossAttnUpBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for SimpleCrossAttnUpBlock2D")
        return SimpleCrossAttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            resnet_time_scale_shift=resnet_time_scale_shift,
            skip_time_act=resnet_skip_time_act,
            output_scale_factor=resnet_out_scale_factor,
            only_cross_attention=only_cross_attention,
            cross_attention_norm=cross_attention_norm,
        )
    elif up_block_type == "AttnUpBlock2D":
        return AttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            attn_num_head_channels=attn_num_head_channels,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "CondAttnUpBlock2D":
        return CondAttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            resnet_time_scale_shift=resnet_time_scale_shift,
            use_gsa=use_gsa,
        )
    elif up_block_type == "SkipUpBlock2D":
        return SkipUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "AttnSkipUpBlock2D":
        return AttnSkipUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            attn_num_head_channels=attn_num_head_channels,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "UpDecoderBlock2D":
        return UpDecoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temb_channels=temb_channels,
        )
    elif up_block_type == "AttnUpDecoderBlock2D":
        return AttnUpDecoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            attn_num_head_channels=attn_num_head_channels,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temb_channels=temb_channels,
        )
    elif up_block_type == "KUpBlock2D":
        return KUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
        )
    elif up_block_type == "KCrossAttnUpBlock2D":
        return KCrossAttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
        )
    raise ValueError(f"{up_block_type} does not exist.")


# def get_xyz_pos(hidden_states):
#     width = hidden_states.shape[-2]
#     height = hidden_states.shape[-1]
#     xy_pos = torch.arange(width).type_as(hidden_states) + 0.5
#     z_pos = torch.arange(height).type_as(hidden_states) + 0.5
#     fov_up = math.pi / 18
#     fov_down = -math.pi / 6

#     pitch = fov_up - z_pos / height * (fov_up - fov_down)
#     xy_pos = math.pi - xy_pos / width * math.pi * 2

#     x_pos = (torch.cos(pitch[None])*torch.cos(xy_pos[:, None]) + 1) / 2
#     y_pos = (torch.cos(pitch[None])*torch.sin(xy_pos[:, None]) + 1) / 2
#     z_pos = (torch.sin(pitch[None]).repeat(width, 1) + 1) / 2
#     xyz_pos = torch.stack([x_pos, y_pos, z_pos], dim=-1)
#     return xyz_pos

class CondAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        cross_attention_dim=768,
        use_gsa=False,
    ):
        super().__init__()
        resnets = []
        attentions = []
        cross_attentions = []
        norms = []
        connectors = []

        if use_gsa:
            attentions2 = []
            norms2 = []
            connectors2 = []
            attn_projs = []

        self.use_gsa = use_gsa
        self.has_cross_attention = True

        # pos_embedders = []
        # self.fourier_embedder = get_embedder(3, 8)

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attn_num_head_channels if attn_num_head_channels is not None else 1,
                    dim_head=attn_num_head_channels if attn_num_head_channels is not None else out_channels,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
            )
            cross_attentions.append(
                Attention(
                    query_dim=out_channels,
                    cross_attention_dim=cross_attention_dim,
                    heads=out_channels // attn_num_head_channels if attn_num_head_channels is not None else 1,
                    dim_head=attn_num_head_channels if attn_num_head_channels is not None else out_channels,
                    norm_num_groups=resnet_groups,
                    residual_connection=False,
                    bias=False,
                ) 
            )
            norms.append(
                nn.LayerNorm(out_channels, elementwise_affine=True)
            )
            connectors.append(zero_module(nn.Linear(out_channels, out_channels)))

            if self.use_gsa:
                connectors2.append(zero_module(nn.Linear(out_channels, out_channels)))
                norms2.append(
                    nn.LayerNorm(out_channels, elementwise_affine=True)
                )
                attentions2.append(
                    Attention(
                        out_channels,
                        heads=out_channels // attn_num_head_channels if attn_num_head_channels is not None else 1,
                        dim_head=attn_num_head_channels if attn_num_head_channels is not None else out_channels,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
                attn_projs.append(
                    nn.Sequential(
                        nn.Linear(cross_attention_dim, out_channels),
                        nn.LayerNorm(out_channels)
                    )
                )

            # pos_embedders.append(
            #     nn.Sequential(
            #         nn.Linear(self.fourier_embedder.out_dim, out_channels),
            #         nn.SiLU(),
            #         zero_module(nn.Linear(out_channels, out_channels))
            #     )
            # )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.cross_attentions = nn.ModuleList(cross_attentions)
        self.connectors = nn.ModuleList(connectors)
        self.norms = nn.ModuleList(norms)

        if self.use_gsa:
            self.attn_projs = nn.ModuleList(attn_projs)
            self.attentions2 = nn.ModuleList(attentions2)
            self.connectors2 = nn.ModuleList(connectors2)
            self.norms2 = nn.ModuleList(norms2)

        # self.pos_embedders = nn.ModuleList(pos_embedders)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

    @property
    def new_module(self):
        block_num = len(self.attentions)
        ret = {}
        for block_id in range(block_num):
            ret[f"cross_attn.{block_id}"] = self.cross_attentions[block_id]
            ret[f"connector.{block_id}"] = self.connectors[block_id]
            ret[f"norm.{block_id}"] = self.norms[block_id]
            # ret[f"pos_embedder.{block_id}"] = self.pos_embedders[block_id]

            if self.use_gsa:
                ret[f"norm2.{block_id}"] = self.norms2[block_id]     
                ret[f"connector2.{block_id}"] = self.connectors2[block_id]     
                ret[f"attention2.{block_id}"] = self.attentions2[block_id]     
                ret[f"attn_proj.{block_id}"] = self.attn_projs[block_id]     

        return ret

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask=None, temb=None, upsample_size=None):
        output_states = ()

        if not self.use_gsa:
            encoder_attention_mask = torch.cat([encoder_attention_mask, encoder_hidden_states[-1]], dim=2)
            encoder_hidden_states = torch.cat(encoder_hidden_states[:2], dim=1) 
        else:
            bbox_attention_mask = encoder_hidden_states[-1]
            bbox_hidden_states = encoder_hidden_states[1]
            encoder_hidden_states = encoder_hidden_states[0]

        block_num = len(self.resnets)
        for block_id in range(block_num):

            resnet = self.resnets[block_id]
            attn = self.attentions[block_id]
            cross_attn = self.cross_attentions[block_id]
            connector = self.connectors[block_id]
            norm = self.norms[block_id]
            # pos_embedder = self.pos_embedders[block_id]

            hidden_states = resnet(hidden_states, temb)

            hidden_states = attn(hidden_states)

            # xyz_pos = get_xyz_pos(hidden_states)
            # xyz_pos = self.fourier_embedder(xyz_pos)
            # xyz_pos = pos_embedder(xyz_pos)
            # xyz_pos = xyz_pos.permute(2, 0, 1)

            # hidden_states = hidden_states + xyz_pos[None]

            hidden_states = hidden_states.permute(0, 2, 3, 1)
            dims = hidden_states.shape[1:3]
            hidden_states = hidden_states.flatten(1,2)

            if self.use_gsa:
                norm2 = self.norms2[block_id]
                connector2 = self.connectors2[block_id]
                attn_proj = self.attn_projs[block_id]
                attn2 = self.attentions2[block_id]
            
                bbox_input = attn_proj(bbox_hidden_states)
                bbox_hidden_mask = torch.cat([bbox_attention_mask, torch.zeros_like(hidden_states[:, None, :, 0])], dim=-1)
                norm_hidden_states = norm2(hidden_states)
                attn_output = attn2(
                    norm_hidden_states,
                    encoder_hidden_states=torch.cat([bbox_input, norm_hidden_states], dim=1),
                    attention_mask=bbox_hidden_mask,
                )
                hidden_states = hidden_states + connector2(attn_output)

            cross_attn_output = cross_attn(
                norm(hidden_states),
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )   
            hidden_states = hidden_states + connector(cross_attn_output)

            hidden_states = hidden_states.permute(0, 2, 1)
            hidden_states = hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[1], dims[0], dims[1])

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class CondAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        add_upsample=True,
        cross_attention_dim=768,
        use_gsa=False,
    ):
        super().__init__()
        resnets = []
        attentions = []
        cross_attentions = []
        norms = []
        connectors = []

        if use_gsa:
            attentions2 = []
            norms2 = []
            connectors2 = []
            attn_projs = []

        self.use_gsa = use_gsa
        self.has_cross_attention = True


        # pos_embedders = []

        self.has_cross_attention = True
        # self.fourier_embedder = get_embedder(3, 8)

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attn_num_head_channels if attn_num_head_channels is not None else 1,
                    dim_head=attn_num_head_channels if attn_num_head_channels is not None else out_channels,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
            )
            cross_attentions.append(
                Attention(
                    query_dim=out_channels,
                    cross_attention_dim=cross_attention_dim,
                    heads=out_channels // attn_num_head_channels if attn_num_head_channels is not None else 1,
                    dim_head=attn_num_head_channels if attn_num_head_channels is not None else out_channels,
                    norm_num_groups=resnet_groups,
                    residual_connection=False,
                    bias=False,
                ) 
            )
            norms.append(
                nn.LayerNorm(out_channels, elementwise_affine=True)
            )
            connectors.append(zero_module(nn.Linear(out_channels, out_channels)))
            # pos_embedders.append(
            #     nn.Sequential(
            #         nn.Linear(self.fourier_embedder.out_dim, out_channels),
            #         nn.SiLU(),
            #         zero_module(nn.Linear(out_channels, out_channels))
            #     )
            # )

            if self.use_gsa:
                connectors2.append(zero_module(nn.Linear(out_channels, out_channels)))
                norms2.append(
                    nn.LayerNorm(out_channels, elementwise_affine=True)
                )
                attentions2.append(
                    Attention(
                        out_channels,
                        heads=out_channels // attn_num_head_channels if attn_num_head_channels is not None else 1,
                        dim_head=attn_num_head_channels if attn_num_head_channels is not None else out_channels,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
                attn_projs.append(
                    nn.Sequential(
                        nn.Linear(cross_attention_dim, out_channels),
                        nn.LayerNorm(out_channels)
                    )
                )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.cross_attentions = nn.ModuleList(cross_attentions)
        self.connectors = nn.ModuleList(connectors)
        self.norms = nn.ModuleList(norms)
        # self.pos_embedders = nn.ModuleList(pos_embedders)

        if self.use_gsa:
            self.attn_projs = nn.ModuleList(attn_projs)
            self.attentions2 = nn.ModuleList(attentions2)
            self.connectors2 = nn.ModuleList(connectors2)
            self.norms2 = nn.ModuleList(norms2)


        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    @property
    def new_module(self):
        block_num = len(self.attentions)
        ret = {}
        for block_id in range(block_num):
            ret[f"cross_attn.{block_id}"] = self.cross_attentions[block_id]
            ret[f"connector.{block_id}"] = self.connectors[block_id]
            ret[f"norm.{block_id}"] = self.norms[block_id]
            # ret[f"pos_embedder.{block_id}"] = self.pos_embedders[block_id]

            if self.use_gsa:
                ret[f"norm2.{block_id}"] = self.norms2[block_id]     
                ret[f"connector2.{block_id}"] = self.connectors2[block_id]     
                ret[f"attention2.{block_id}"] = self.attentions2[block_id]     
                ret[f"attn_proj.{block_id}"] = self.attn_projs[block_id]     

        return ret

    def forward(self, hidden_states, res_hidden_states_tuple, encoder_hidden_states, encoder_attention_mask=None, temb=None, upsample_size=None):

        if not self.use_gsa:
            encoder_attention_mask = torch.cat([encoder_attention_mask, encoder_hidden_states[-1]], dim=2)
            encoder_hidden_states = torch.cat(encoder_hidden_states[:2], dim=1) 
        else:
            bbox_attention_mask = encoder_hidden_states[-1]
            bbox_hidden_states = encoder_hidden_states[1]
            encoder_hidden_states = encoder_hidden_states[0]

        block_num = len(self.resnets)
        for block_id in range(block_num):
            resnet = self.resnets[block_id]
            attn = self.attentions[block_id]
            cross_attn = self.cross_attentions[block_id]
            connector = self.connectors[block_id]
            norm = self.norms[block_id]
            # pos_embedder = self.pos_embedders[block_id]

            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)

            # xyz_pos = get_xyz_pos(hidden_states)
            # xyz_pos = self.fourier_embedder(xyz_pos)
            # xyz_pos = pos_embedder(xyz_pos)
            # xyz_pos = xyz_pos.permute(2, 0, 1)

            # hidden_states = hidden_states + xyz_pos[None]

            hidden_states = hidden_states.permute(0, 2, 3, 1)
            dims = hidden_states.shape[1:3]
            hidden_states = hidden_states.flatten(1,2)

            if self.use_gsa:
                norm2 = self.norms2[block_id]
                connector2 = self.connectors2[block_id]
                attn_proj = self.attn_projs[block_id]
                attn2 = self.attentions2[block_id]
            
                bbox_input = attn_proj(bbox_hidden_states)
                bbox_hidden_mask = torch.cat([bbox_attention_mask, torch.zeros_like(hidden_states[:, None, :, 0])], dim=-1)
                norm_hidden_states = norm2(hidden_states)
                attn_output = attn2(
                    norm_hidden_states,
                    encoder_hidden_states=torch.cat([bbox_input, norm_hidden_states], dim=1),
                    attention_mask=bbox_hidden_mask,
                )
                hidden_states = hidden_states + connector2(attn_output)

            cross_attn_output = cross_attn(
                norm(hidden_states),
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )   
            hidden_states = hidden_states + connector(cross_attn_output)

            hidden_states = hidden_states.permute(0, 2, 1)
            hidden_states = hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[1], dims[0], dims[1])

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class UNetMidBlock2DCond(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=768,
        use_gsa=False,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []
        cross_attentions = []
        norms = []
        connectors = []

        if use_gsa:
            attentions2 = []
            norms2 = []
            connectors2 = []
            attn_projs = []

        self.use_gsa = use_gsa
        # pos_embedders = []

        self.has_cross_attention = True
        # self.fourier_embedder = get_embedder(3, 8)

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attn_num_head_channels if attn_num_head_channels is not None else 1,
                        dim_head=attn_num_head_channels if attn_num_head_channels is not None else in_channels,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups if resnet_time_scale_shift == "default" else None,
                        spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
                cross_attentions.append(
                    Attention(
                        query_dim=in_channels,
                        cross_attention_dim=cross_attention_dim,
                        heads=in_channels // attn_num_head_channels if attn_num_head_channels is not None else 1,
                        dim_head=attn_num_head_channels if attn_num_head_channels is not None else in_channels,
                        norm_num_groups=resnet_groups,
                        residual_connection=False,
                        bias=False,
                    ) 
                )
                norms.append(
                    nn.LayerNorm(in_channels, elementwise_affine=True)
                )
                connectors.append(zero_module(nn.Linear(in_channels, in_channels)))
                # pos_embedders.append(
                #     nn.Sequential(
                #         nn.Linear(self.fourier_embedder.out_dim, in_channels),
                #         nn.SiLU(),
                #         zero_module(nn.Linear(in_channels, in_channels))
                #     )
                # )
            else:
                attentions.append(None)
                cross_attentions.append(None)
                norms.append(None)
                connectors.append(None)
                # pos_embedders.append(None)

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

            if self.use_gsa:
                connectors2.append(zero_module(nn.Linear(in_channels, in_channels)))
                norms2.append(
                    nn.LayerNorm(in_channels, elementwise_affine=True)
                )
                attentions2.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attn_num_head_channels if attn_num_head_channels is not None else 1,
                        dim_head=attn_num_head_channels if attn_num_head_channels is not None else in_channels,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
                attn_projs.append(
                    nn.Sequential(
                        nn.Linear(cross_attention_dim, in_channels),
                        nn.LayerNorm(in_channels)
                    )
                )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.cross_attentions = nn.ModuleList(cross_attentions)
        self.connectors = nn.ModuleList(connectors)
        self.norms = nn.ModuleList(norms)
        # self.pos_embedders = nn.ModuleList(pos_embedders)

        if self.use_gsa:
            self.attn_projs = nn.ModuleList(attn_projs)
            self.attentions2 = nn.ModuleList(attentions2)
            self.connectors2 = nn.ModuleList(connectors2)
            self.norms2 = nn.ModuleList(norms2)

    @property
    def new_module(self):
        block_num = len(self.attentions)
        ret = {}
        for block_id in range(block_num):
            ret[f"cross_attn.{block_id}"] = self.cross_attentions[block_id]
            ret[f"connector.{block_id}"] = self.connectors[block_id]
            ret[f"norm.{block_id}"] = self.norms[block_id]
            # ret[f"pos_embedder.{block_id}"] = self.pos_embedders[block_id]

            if self.use_gsa:
                ret[f"norm2.{block_id}"] = self.norms2[block_id]     
                ret[f"connector2.{block_id}"] = self.connectors2[block_id]     
                ret[f"attention2.{block_id}"] = self.attentions2[block_id]     
                ret[f"attn_proj.{block_id}"] = self.attn_projs[block_id]     

        return ret

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask=None, temb=None):
        if not self.use_gsa:
            encoder_attention_mask = torch.cat([encoder_attention_mask, encoder_hidden_states[-1]], dim=2)
            encoder_hidden_states = torch.cat(encoder_hidden_states[:2], dim=1) 
        else:
            bbox_attention_mask = encoder_hidden_states[-1]
            bbox_hidden_states = encoder_hidden_states[1]
            encoder_hidden_states = encoder_hidden_states[0]

        hidden_states = self.resnets[0](hidden_states, temb)
        block_num = len(self.attentions)
        for block_id in range(block_num):
            attn = self.attentions[block_id]
            resnet = self.resnets[block_id+1]
            if attn is not None:
                hidden_states = attn(hidden_states, temb=temb)


            norm = self.norms[block_id]
            connector = self.connectors[block_id]
            cross_attn = self.cross_attentions[block_id]
            # pos_embedder = self.pos_embedders[block_id]
            if cross_attn is not None:
                # xyz_pos = get_xyz_pos(hidden_states)
                # xyz_pos = self.fourier_embedder(xyz_pos)
                # xyz_pos = pos_embedder(xyz_pos)
                # xyz_pos = xyz_pos.permute(2, 0, 1)

                # hidden_states = hidden_states + xyz_pos[None]

                hidden_states = hidden_states.permute(0, 2, 3, 1)
                dims = hidden_states.shape[1:3]
                hidden_states = hidden_states.flatten(1,2)

                if self.use_gsa:
                    norm2 = self.norms2[block_id]
                    connector2 = self.connectors2[block_id]
                    attn_proj = self.attn_projs[block_id]
                    attn2 = self.attentions2[block_id]
                
                    bbox_input = attn_proj(bbox_hidden_states)
                    bbox_hidden_mask = torch.cat([bbox_attention_mask, torch.zeros_like(hidden_states[:, None, :, 0])], dim=-1)
                    norm_hidden_states = norm2(hidden_states)
                    attn_output = attn2(
                        norm_hidden_states,
                        encoder_hidden_states=torch.cat([bbox_input, norm_hidden_states], dim=1),
                        attention_mask=bbox_hidden_mask,
                    )
                    hidden_states = hidden_states + connector2(attn_output)

                cross_attn_output = cross_attn(
                    norm(hidden_states),
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                )   
                hidden_states = hidden_states + connector(cross_attn_output)

                hidden_states = hidden_states.permute(0, 2, 1)
                hidden_states = hidden_states.reshape(hidden_states.shape[0], hidden_states.shape[1], dims[0], dims[1])

            hidden_states = resnet(hidden_states, temb)

        return hidden_states
