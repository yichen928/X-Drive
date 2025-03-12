import logging

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .embedder import get_embedder

from diffusers.models.modeling_utils import ModelMixin
from diffusers import ConfigMixin

XYZ_MIN = [-51.2, -51.2, -3.]
XYZ_RANGE = [102.4, 102.4, 8]

XYZD_MIN = [-51.2, -51.2, -3., 0]
XYZD_RANGE = [102.4, 102.4, 8, 80]

def normalizer(mode, data):
    if mode == 'cxyz' or mode == 'all-xyz':
        # data in format of (N, 4, 3):
        mins = torch.as_tensor(
            XYZ_MIN, dtype=data.dtype, device=data.device)[None, None]
        divider = torch.as_tensor(
            XYZ_RANGE, dtype=data.dtype, device=data.device)[None, None]
        data = (data - mins) / divider
    elif mode == 'all-xyzd':
        mins = torch.as_tensor(
            XYZD_MIN, dtype=data.dtype, device=data.device)[None, None]
        divider = torch.as_tensor(
            XYZD_RANGE, dtype=data.dtype, device=data.device)[None, None]
        data = (data - mins) / divider      
    elif mode == 'owhr':
        raise NotImplementedError(f"wait for implementation on {mode}")
    else:
        raise NotImplementedError(f"not support {mode}")
    return data


class ContinuousBBoxWithTextEmbedding(ModelMixin, ConfigMixin):
    """
    Use continuous bbox corrdicate and text embedding with CLIP encoder
    """

    def __init__(
        self,
        n_classes,
        class_token_dim=768,
        trainable_class_token=False,
        embedder_num_freq=4,
        proj_dims=[768, 512, 512, 768],
        mode='cxyz',
        minmax_normalize=True,
        use_text_encoder_init=True,
        **kwargs,
    ):
        """
        Args:
            mode (str, optional): cxyz -> all points; all-xyz -> all points;
                owhr -> center, l, w, h, z-orientation.
        """
        super().__init__()

        self.mode = mode
        if self.mode == 'cxyz':
            input_dims = 3
            output_num = 4  # 4 points
        elif self.mode == 'all-xyz':
            input_dims = 3
            output_num = 8  # 8 points
        elif self.mode == 'owhr':
            raise NotImplementedError("Not sure how to do this.")
        else:
            raise NotImplementedError(f"Wrong mode {mode}")
        self.minmax_normalize = minmax_normalize
        self.use_text_encoder_init = use_text_encoder_init

        self.fourier_embedder = get_embedder(input_dims, embedder_num_freq)
        logging.info(
            f"[ContinuousBBoxWithTextEmbedding] bbox embedder has "
            f"{self.fourier_embedder.out_dim} dims.")

        self.bbox_proj = nn.Linear(
            self.fourier_embedder.out_dim * output_num, proj_dims[0])
        self.second_linear = nn.Sequential(
            nn.Linear(proj_dims[0] + class_token_dim, proj_dims[1]),
            nn.SiLU(),
            nn.Linear(proj_dims[1], proj_dims[2]),
            nn.SiLU(),
            nn.Linear(proj_dims[2], proj_dims[3]),
        )

        # for class token
        self._class_tokens_set_or_warned = not self.use_text_encoder_init
        if trainable_class_token:
            # parameter is trainable, buffer is not
            class_tokens = torch.randn(n_classes, class_token_dim)
            self.register_parameter("_class_tokens", nn.Parameter(class_tokens))
        else:
            class_tokens = torch.randn(n_classes, class_token_dim)
            self.register_buffer("_class_tokens", class_tokens)
            if not self.use_text_encoder_init:
                logging.warn(
                    "[ContinuousBBoxWithTextEmbedding] Your class_tokens is not"
                    " trainable but you set `use_text_encoder_init` to False. "
                    "Please check your config!")

        # null embedding
        self.null_class_feature = torch.nn.Parameter(
            torch.zeros([class_token_dim]))
        self.null_pos_feature = torch.nn.Parameter(
            torch.zeros([self.fourier_embedder.out_dim * output_num]))

    @property
    def class_tokens(self):
        if not self._class_tokens_set_or_warned:
            logging.warn(
                "[ContinuousBBoxWithTextEmbedding] Your class_tokens is not "
                "trainable and used without initialization. Please check your "
                "training code!")
            self._class_tokens_set_or_warned = True
        return self._class_tokens

    def prepare(self, cfg, **kwargs):
        if self.use_text_encoder_init:
            self.set_category_token(
                kwargs['tokenizer'], kwargs['text_encoder'],
                cfg.dataset.object_classes)
        else:
            logging.info("[ContinuousBBoxWithTextEmbedding] Your class_tokens "
                         "initilzed with random.")

    @torch.no_grad()
    def set_category_token(self, tokenizer, text_encoder, class_names):
        logging.info("[ContinuousBBoxWithTextEmbedding] Initialzing your "
                     "class_tokens with text_encoder")
        self._class_tokens_set_or_warned = True
        device = self.class_tokens.device
        for idx, name in enumerate(class_names):
            inputs = tokenizer(
                [name], padding='do_not_pad', return_tensors='pt')
            inputs = inputs.input_ids.to(device)
            # there are two outputs: last_hidden_state and pooler_output
            # we use the pooled version.
            hidden_state = text_encoder(inputs).pooler_output[0]  # 768
            self.class_tokens[idx].copy_(hidden_state)

    def add_n_uncond_tokens(self, hidden_states, token_num):
        B = hidden_states.shape[0]
        uncond_token = self.forward_feature(
            self.null_pos_feature[None], self.null_class_feature[None])
        uncond_token = repeat(uncond_token, 'c -> b n c', b=B, n=token_num)
        hidden_states = torch.cat([hidden_states, uncond_token], dim=1)
        return hidden_states

    def forward_feature(self, pos_emb, cls_emb):
        emb = self.bbox_proj(pos_emb)
        emb = F.silu(emb)

        # combine
        emb = torch.cat([emb, cls_emb], dim=-1)
        emb = self.second_linear(emb)
        return emb

    def forward(self, bboxes: torch.Tensor, classes: torch.LongTensor,
                masks=None, **kwargs):
        """Please do filter before input is needed.

        Args:
            bboxes (torch.Tensor): Expect (B, N, 4, 3) for cxyz mode.
            classes (torch.LongTensor): (B, N)

        Return:
            size B x N x emb_dim=768
        """
        (B, N) = classes.shape
        bboxes = rearrange(bboxes, 'b n ... -> (b n) ...')

        if masks is None:
            masks = torch.ones(len(bboxes))
        else:
            masks = masks.flatten()
        masks = masks.unsqueeze(-1).type_as(self.null_pos_feature)

        # box
        bboxes = normalizer(self.mode, bboxes)
        pos_emb = self.fourier_embedder(bboxes)
        pos_emb = pos_emb.reshape(
            pos_emb.shape[0], -1).type_as(self.null_pos_feature)
        pos_emb = pos_emb * masks + self.null_pos_feature[None] * (1 - masks)

        # class
        cls_emb = torch.stack([self.class_tokens[i] for i in classes.flatten()])
        cls_emb = cls_emb * masks + self.null_class_feature[None] * (1 - masks)

        # combine
        emb = self.forward_feature(pos_emb, cls_emb)
        emb = rearrange(emb, '(b n) ... -> b n ...', n=N)
        return emb


class ContinuousBBoxViewWithTextEmbedding(ModelMixin, ConfigMixin):
    """
    Use continuous bbox coordinate and text embedding with CLIP encoder
    """

    def __init__(
        self,
        n_classes,
        class_token_dim=768,
        trainable_class_token=False,
        embedder_num_freq=4,
        proj_dims=[768, 512, 512, 768],
        minmax_normalize=True,
        use_text_encoder_init=True,
        canvas_size=(400, 224),
        mode="all-wh",
        max_depth=75,
        **kwargs,
    ):
        """
        Args:
            mode (str, optional): cxyz -> all points; all-xyz -> all points;
                owhr -> center, l, w, h, z-orientation.
        """
        super().__init__()

        self.canvas_size = canvas_size
        self.mode = mode
        self.max_depth = max_depth
        if self.mode == "all-wh":
            input_dims = 2
            output_num = 8  # 4 points
        else:
            input_dims = 3
            output_num = 8  # 4 points     

        self.minmax_normalize = minmax_normalize
        self.use_text_encoder_init = use_text_encoder_init

        self.fourier_embedder = get_embedder(input_dims, embedder_num_freq)
        logging.info(
            f"[ContinuousBBoxWithTextEmbedding] bbox embedder has "
            f"{self.fourier_embedder.out_dim} dims.")

        self.bbox_proj = nn.Linear(
            self.fourier_embedder.out_dim * output_num, proj_dims[0])
        self.second_linear = nn.Sequential(
            nn.Linear(proj_dims[0] + class_token_dim, proj_dims[1]),
            nn.SiLU(),
            nn.Linear(proj_dims[1], proj_dims[2]),
            nn.SiLU(),
            nn.Linear(proj_dims[2], proj_dims[3]),
        )

        # for class token
        self._class_tokens_set_or_warned = not self.use_text_encoder_init
        if trainable_class_token:
            # parameter is trainable, buffer is not
            class_tokens = torch.randn(n_classes, class_token_dim)
            self.register_parameter("_class_tokens", nn.Parameter(class_tokens))
        else:
            class_tokens = torch.randn(n_classes, class_token_dim)
            self.register_buffer("_class_tokens", class_tokens)
            if not self.use_text_encoder_init:
                logging.warn(
                    "[ContinuousBBoxWithTextEmbedding] Your class_tokens is not"
                    " trainable but you set `use_text_encoder_init` to False. "
                    "Please check your config!")

        # null embedding
        self.null_class_feature = torch.nn.Parameter(
            torch.zeros([class_token_dim]))
        self.null_pos_feature = torch.nn.Parameter(
            torch.zeros([self.fourier_embedder.out_dim * output_num]))

    @property
    def class_tokens(self):
        if not self._class_tokens_set_or_warned:
            logging.warn(
                "[ContinuousBBoxWithTextEmbedding] Your class_tokens is not "
                "trainable and used without initialization. Please check your "
                "training code!")
            self._class_tokens_set_or_warned = True
        return self._class_tokens

    def prepare(self, cfg, **kwargs):
        if self.use_text_encoder_init:
            self.set_category_token(
                kwargs['tokenizer'], kwargs['text_encoder'],
                cfg.dataset.object_classes)
        else:
            logging.info("[ContinuousBBoxWithTextEmbedding] Your class_tokens "
                         "initilzed with random.")

    @torch.no_grad()
    def set_category_token(self, tokenizer, text_encoder, class_names):
        logging.info("[ContinuousBBoxWithTextEmbedding] Initialzing your "
                     "class_tokens with text_encoder")
        self._class_tokens_set_or_warned = True
        device = self.class_tokens.device
        for idx, name in enumerate(class_names):
            inputs = tokenizer(
                [name], padding='do_not_pad', return_tensors='pt')
            inputs = inputs.input_ids.to(device)
            # there are two outputs: last_hidden_state and pooler_output
            # we use the pooled version.
            hidden_state = text_encoder(inputs).pooler_output[0]  # 768
            self.class_tokens[idx].copy_(hidden_state)

    def add_n_uncond_tokens(self, hidden_states, token_num):
        B = hidden_states.shape[0]
        uncond_token = self.forward_feature(
            self.null_pos_feature[None], self.null_class_feature[None])
        uncond_token = repeat(uncond_token, 'c -> b n c', b=B, n=token_num)
        hidden_states = torch.cat([hidden_states, uncond_token], dim=1)
        return hidden_states

    def n_uncond_tokens(self, hidden_states, token_num):
        B = hidden_states.shape[0]
        uncond_token = self.forward_feature(
            self.null_pos_feature[None], self.null_class_feature[None])[0]
        uncond_token = repeat(uncond_token, 'c -> b n c', b=B, n=token_num)
        return uncond_token

    def forward_feature(self, pos_emb, cls_emb):
        emb = self.bbox_proj(pos_emb)
        emb = F.silu(emb)

        # combine
        emb = torch.cat([emb, cls_emb], dim=-1)
        emb = self.second_linear(emb)
        return emb

    def project_bboxes_to_images(self, bboxes, masks, lidar2imgs, normalize=True):
        bs = bboxes.shape[0]
        lidar2imgs = lidar2imgs.view(bs, 4, 4)
        bboxes_pad = torch.cat([bboxes, torch.ones_like(bboxes[...,:1])], dim=-1)
        bboxes_view = torch.matmul(lidar2imgs[:, None, None], bboxes_pad[..., None])
        bboxes_view = bboxes_view.squeeze(-1)[..., :3]
        bboxes_depth = torch.where((bboxes_view[..., 2:3]<1)&(bboxes_view[..., 2:3]>=0), torch.ones_like(bboxes_view[..., 2:3]), bboxes_view[..., 2:3])
        bboxes_depth = torch.where((bboxes_view[..., 2:3]>-1)&(bboxes_view[..., 2:3]<0), -torch.ones_like(bboxes_view[..., 2:3]), bboxes_view[..., 2:3])
        bboxes_view[..., :2] = bboxes_view[..., :2] / bboxes_depth

        view_mask = (bboxes_view[..., 2] > 0) & (bboxes_view[..., 0] > 0) & (bboxes_view[..., 0] < self.canvas_size[0]) & (bboxes_view[..., 1] > 0) & (bboxes_view[..., 1] < self.canvas_size[1])
        view_mask = torch.max(view_mask, dim=-1)[0]

        masks = torch.logical_and(view_mask, masks)
        bboxes = bboxes_view[..., :2]
        bboxes_d = bboxes_view[..., 2:3]
        if normalize:
            bboxes[..., 0] = bboxes[..., 0] / self.canvas_size[0]
            bboxes[..., 1] = bboxes[..., 1] / self.canvas_size[1]
            bboxes_d = bboxes_d / self.max_depth

        # bboxes_max = torch.max(bboxes.flatten(2,3), dim=2)[0]
        # bboxes_min = torch.min(bboxes.flatten(2,3), dim=2)[0]
        # masks = masks & (bboxes_max < 2.0) & (bboxes_min > -1.0)
        bboxes = torch.clamp(bboxes, min=-1.0, max=2.0)

        if self.mode == "all-whd":
            bboxes = torch.cat([bboxes, bboxes_d], dim=-1)

        bboxes = torch.where(masks[..., None, None], bboxes, torch.zeros_like(bboxes))

        assert torch.isnan(bboxes).sum() == 0

        return bboxes, masks

    def forward(self, bboxes: torch.Tensor, classes: torch.LongTensor,
                lidar2imgs: torch.Tensor, masks=None, **kwargs):
        """lease do filter before input is needed.

        Args:
            bboxes (torch.Tensor): Expect (B, N, 4, 3) for cxyz mode.
            classes (torch.LongTensor): (B, N)

        Return:
            size B x N x emb_dim=768
        """

        bboxes, bboxes_masks = self.project_bboxes_to_images(bboxes, masks, lidar2imgs)
        (B, N) = classes.shape
        bboxes = rearrange(bboxes, 'b n ... -> (b n) ...')

        masks = bboxes_masks
        if masks is None:
            masks = torch.ones(len(bboxes))
        else:
            masks = masks.flatten()
        masks = masks.unsqueeze(-1).type_as(self.null_pos_feature)

        # box
        if self.minmax_normalize:
            bboxes = normalizer(self.mode, bboxes)
        assert torch.isnan(bboxes).sum() == 0

        pos_emb = self.fourier_embedder(bboxes)
        assert torch.isnan(pos_emb).sum() == 0

        pos_emb = pos_emb.reshape(
            pos_emb.shape[0], -1).type_as(self.null_pos_feature)
        pos_emb = pos_emb * masks + self.null_pos_feature[None] * (1 - masks)

        # class
        cls_emb = torch.stack([self.class_tokens[i] for i in classes.flatten()])
        cls_emb = cls_emb * masks + self.null_class_feature[None] * (1 - masks)

        # combine
        emb = self.forward_feature(pos_emb, cls_emb)
        emb = rearrange(emb, '(b n) ... -> b n ...', n=N)
        assert torch.isnan(pos_emb).sum() == 0
        assert torch.isnan(cls_emb).sum() == 0

        return emb, bboxes_masks


class ContinuousBBoxXYZWithTextEmbedding(ModelMixin, ConfigMixin):
    """
    Use continuous bbox coordinate and text embedding with CLIP encoder
    """

    def __init__(
        self,
        n_classes,
        class_token_dim=768,
        trainable_class_token=False,
        embedder_num_freq=4,
        proj_dims=[768, 512, 512, 768],
        mode='cxyz',
        minmax_normalize=True,
        use_text_encoder_init=True,
        fov=(-30, 10),
        max_range=80,
        **kwargs,
    ):
        """
        Args:
            mode (str, optional): cxyz -> all points; all-xyz -> all points;
                owhr -> center, l, w, h, z-orientation.
        """
        super().__init__()

        self.mode = mode
        if self.mode == 'cxyz':
            input_dims = 3
            output_num = 4  # 4 points
        elif self.mode == 'all-xyz':
            input_dims = 3
            output_num = 8  # 8 points
        elif self.mode == 'all-xyzd':
            input_dims = 4
            output_num = 8
            
        elif self.mode == 'owhr':
            raise NotImplementedError("Not sure how to do this.")
        else:
            raise NotImplementedError(f"Wrong mode {mode}")

        self.fov = (fov[0]/180*math.pi, fov[1]/180*math.pi)
        self.max_range = max_range
        self.minmax_normalize = minmax_normalize
        self.use_text_encoder_init = use_text_encoder_init

        self.fourier_embedder = get_embedder(input_dims, embedder_num_freq)
        logging.info(
            f"[ContinuousBBoxWithTextEmbedding] bbox embedder has "
            f"{self.fourier_embedder.out_dim} dims.")

        self.bbox_proj = nn.Linear(
            self.fourier_embedder.out_dim * output_num, proj_dims[0])
        self.second_linear = nn.Sequential(
            nn.Linear(proj_dims[0] + class_token_dim, proj_dims[1]),
            nn.SiLU(),
            nn.Linear(proj_dims[1], proj_dims[2]),
            nn.SiLU(),
            nn.Linear(proj_dims[2], proj_dims[3]),
        )

        # for class token
        self._class_tokens_set_or_warned = not self.use_text_encoder_init
        if trainable_class_token:
            # parameter is trainable, buffer is not
            class_tokens = torch.randn(n_classes, class_token_dim)
            self.register_parameter("_class_tokens", nn.Parameter(class_tokens))
        else:
            class_tokens = torch.randn(n_classes, class_token_dim)
            self.register_buffer("_class_tokens", class_tokens)
            if not self.use_text_encoder_init:
                logging.warn(
                    "[ContinuousBBoxWithTextEmbedding] Your class_tokens is not"
                    " trainable but you set `use_text_encoder_init` to False. "
                    "Please check your config!")

        # null embedding
        self.null_class_feature = torch.nn.Parameter(
            torch.zeros([class_token_dim]))
        self.null_pos_feature = torch.nn.Parameter(
            torch.zeros([self.fourier_embedder.out_dim * output_num]))

    @property
    def class_tokens(self):
        if not self._class_tokens_set_or_warned:
            logging.warn(
                "[ContinuousBBoxWithTextEmbedding] Your class_tokens is not "
                "trainable and used without initialization. Please check your "
                "training code!")
            self._class_tokens_set_or_warned = True
        return self._class_tokens

    def prepare(self, cfg, **kwargs):
        if self.use_text_encoder_init:
            self.set_category_token(
                kwargs['tokenizer'], kwargs['text_encoder'],
                cfg.dataset.object_classes)
        else:
            logging.info("[ContinuousBBoxWithTextEmbedding] Your class_tokens "
                         "initilzed with random.")

    @torch.no_grad()
    def set_category_token(self, tokenizer, text_encoder, class_names):
        logging.info("[ContinuousBBoxWithTextEmbedding] Initialzing your "
                     "class_tokens with text_encoder")
        self._class_tokens_set_or_warned = True
        device = self.class_tokens.device
        for idx, name in enumerate(class_names):
            inputs = tokenizer(
                [name], padding='do_not_pad', return_tensors='pt')
            inputs = inputs.input_ids.to(device)
            # there are two outputs: last_hidden_state and pooler_output
            # we use the pooled version.
            hidden_state = text_encoder(inputs).pooler_output[0]  # 768
            self.class_tokens[idx].copy_(hidden_state)

    def add_n_uncond_tokens(self, hidden_states, token_num):
        B = hidden_states.shape[0]
        uncond_token = self.forward_feature(
            self.null_pos_feature[None], self.null_class_feature[None])
        uncond_token = repeat(uncond_token, 'c -> b n c', b=B, n=token_num)
        hidden_states = torch.cat([hidden_states, uncond_token], dim=1)
        return hidden_states

    def n_uncond_tokens(self, hidden_states, token_num):
        B = hidden_states.shape[0]
        uncond_token = self.forward_feature(
            self.null_pos_feature[None], self.null_class_feature[None])[0]
        uncond_token = repeat(uncond_token, 'c -> b n c', b=B, n=token_num)
        return uncond_token

    def forward_feature(self, pos_emb, cls_emb):
        emb = self.bbox_proj(pos_emb)
        emb = F.silu(emb)

        # combine
        emb = torch.cat([emb, cls_emb], dim=-1)
        emb = self.second_linear(emb)
        return emb

    def forward(self, bboxes: torch.Tensor, classes: torch.LongTensor,
                masks=None, **kwargs):
        """Please do filter before input is needed.

        Args:
            bboxes (torch.Tensor): Expect (B, N, 4, 3) for cxyz mode.
            classes (torch.LongTensor): (B, N)

        Return:
            size B x N x emb_dim=768
        """

        if self.mode == 'all-xyzd':
            bboxes_range = torch.norm(bboxes, dim=-1, keepdim=True)
            bboxes = torch.cat([bboxes, bboxes_range], dim=-1)
        (B, N) = classes.shape
        bboxes = rearrange(bboxes, 'b n ... -> (b n) ...')

        if masks is None:
            masks = torch.ones(len(bboxes))
        else:
            masks = masks.flatten()
        masks = masks.unsqueeze(-1).type_as(self.null_pos_feature)

        # box
        if self.minmax_normalize:
            bboxes = normalizer(self.mode, bboxes)
        import pdb
        pdb.set_trace
        pos_emb = self.fourier_embedder(bboxes)
        pos_emb = pos_emb.reshape(
            pos_emb.shape[0], -1).type_as(self.null_pos_feature)
        pos_emb = pos_emb * masks + self.null_pos_feature[None] * (1 - masks)

        # class
        cls_emb = torch.stack([self.class_tokens[i] for i in classes.flatten()])
        cls_emb = cls_emb * masks + self.null_class_feature[None] * (1 - masks)

        # combine
        emb = self.forward_feature(pos_emb, cls_emb)
        emb = rearrange(emb, '(b n) ... -> b n ...', n=N)

        return emb