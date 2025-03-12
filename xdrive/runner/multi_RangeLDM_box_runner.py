import os
import math
import logging
from omegaconf import OmegaConf
from functools import partial
from packaging import version
from tqdm.auto import tqdm

import safetensors
import random
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from diffusers.training_utils import EMAModel

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
    UNet2DConditionModel
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from accelerate import Accelerator, DistributedType

from xdrive.dataset.utils import multi_bbox_collate_fn
from xdrive.runner.multi_validator import MultiValidatorBox
from xdrive.runner.utils import (
    prepare_ckpt,
    resume_all_scheduler,
    append_dims,
)
from xdrive.misc.common import (
    load_module,
    deepspeed_zero_init_disabled_context_manager,
)

from ..misc.common import load_module, convert_outputs_to_fp16
from .base_runner import BaseRunner

from xdrive.networks.unet_2d_sep_condition_multiview import UNet2DSepConditionModelMultiview
from third_party.RangeLDM.ldm.utils import replace_attn, replace_conv, replace_down
from xdrive.networks.unet_pc_condition_RangeLDM import RangeLDMPCUNet2DModel

class MultiRangeLDMBoxRunner(BaseRunner):
    def __init__(self, cfg, accelerator, train_set, val_set) -> None:
        # super().__init__(cfg, accelerator, train_set, val_set)
        self.cfg = cfg
        self.accelerator: Accelerator = accelerator
        # Load models and create wrapper for stable diffusion
        # workaround for ZeRO-3, see:
        # https://github.com/huggingface/diffusers/blob/3ebbaf7c96801271f9e6c21400033b6aa5ffcf29/examples/text_to_image/train_text_to_image.py#L571
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            self._init_fixed_models(cfg)
        self._init_trainable_models(cfg)

        # set model and xformers
        self._set_model_trainable_state()
        self._set_xformer_state()
        self._set_gradient_checkpointing()
        
        self.use_ema = self.cfg.runner.use_ema

        # dataloaders
        self.train_dataset = train_set
        self.train_dataloader = None
        self.val_dataset = val_set
        self.val_dataloader = None
        self._set_dataset_loader()

        # param and placeholders
        self.weight_dtype = torch.float32
        self.overrode_max_train_steps = self.cfg.runner.max_train_steps is None
        self.num_update_steps_per_epoch = None  # based on train loader
        self.optimizer = None
        self.lr_scheduler = None


        pipe_cls = load_module(self.cfg.model.pipe_module)
        self.validator = MultiValidatorBox(
            self.cfg,
            self.val_dataset,
            pipe_cls,
            pipe_param={
                "vae_pc": self.pc_vae,
                "vae_img": self.img_vae,
                "text_encoder": self.text_encoder,
                "tokenizer": self.tokenizer,
                # "scheduler_pc": DDPMScheduler.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="scheduler"),
                "scheduler_pc": DDPMScheduler.from_pretrained(cfg.model.pretrained_pc_model_path, subfolder="scheduler"),
                "scheduler_img": DDPMScheduler.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="scheduler"),
            }
        )

    def _init_fixed_models(self, cfg):
        # fmt: off
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler_img = DDPMScheduler.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler_pc = DDPMScheduler.from_pretrained(cfg.model.pretrained_pc_model_path, subfolder="scheduler")


        pc_vae_config_path = os.path.join(cfg.model.pretrained_pc_model_path, "vae", "config.json")
        pc_vae_checkpoint_path = os.path.join(cfg.model.pretrained_pc_model_path, "vae", "diffusion_pytorch_model.safetensors")

        pc_vae_config = AutoencoderKL.load_config(pc_vae_config_path)
        pc_vae = AutoencoderKL.from_config(pc_vae_config)
        pc_vae_checkpoint = safetensors.torch.load_file(pc_vae_checkpoint_path)
        if 'quant_conv.weight' not in pc_vae_checkpoint:
            pc_vae.quant_conv = torch.nn.Identity()
            pc_vae.post_quant_conv = torch.nn.Identity()
        replace_down(pc_vae)
        replace_conv(pc_vae)
        if 'encoder.mid_block.attentions.0.to_q.weight' not in pc_vae_checkpoint:
            replace_attn(pc_vae)
        pc_vae.load_state_dict(pc_vae_checkpoint)

        self.pc_vae = pc_vae

        self.img_vae = AutoencoderKL.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="vae")
        # fmt: on

    def _init_trainable_models(self, cfg):
    
        bbox_embedder_ckpt_dict = None
        model_cls = load_module(self.cfg.model.pc_bbox_embedder_cls)
        self.pc_bbox_embedder = model_cls(**self.cfg.model.pc_bbox_embedder_param)
        self.pc_bbox_embedder.prepare(self.cfg, **{"tokenizer": self.tokenizer, "text_encoder": self.text_encoder})

        if bbox_embedder_ckpt_dict is not None:
            self.pc_bbox_embedder.load_state_dict(bbox_embedder_ckpt_dict)

        model_cls = load_module(self.cfg.model.img_bbox_embedder_cls)
        self.img_bbox_embedder = model_cls(**self.cfg.model.img_bbox_embedder_param)
        self.img_bbox_embedder.prepare(self.cfg, **{"tokenizer": self.tokenizer, "text_encoder": self.text_encoder})

        # pc_unet_class = load_module(self.cfg.model.pc_unet_class)
        # pc_unet = pc_unet_class(**cfg.model.pc_unet)
        
        # if pc_unet_ckpt_dict is not None:
        #     pc_unet.load_state_dict(pc_unet_ckpt_dict, strict=False)

        pc_unet_checkpoint_path = os.path.join(cfg.model.pretrained_pc_model_path, "unet", "diffusion_pytorch_model.safetensors")
        pc_unet_config_path = os.path.join(cfg.model.pretrained_pc_model_path, "unet", "config.json")

        pc_unet_config = RangeLDMPCUNet2DModel.load_config(pc_unet_config_path)
        pc_unet_config['down_block_types'] = ['DownBlock2D', 'CondAttnDownBlock2D', 'CondAttnDownBlock2D', 'CondAttnDownBlock2D']
        pc_unet_config['up_block_types'] = ['CondAttnUpBlock2D', 'CondAttnUpBlock2D', 'CondAttnUpBlock2D', 'UpBlock2D']
        pc_unet_config['cross_attention_dim'] = cfg.model.pc_unet.cross_attention_dim
        pc_unet_config['trainable_state'] = cfg.model.pc_unet.trainable_state
        pc_unet_config['use_gsa'] = cfg.model.pc_unet.use_gsa

        pc_unet = RangeLDMPCUNet2DModel.from_config(pc_unet_config)

        replace_down(pc_unet)
        replace_conv(pc_unet)
        safetensors.torch.load_model(pc_unet, pc_unet_checkpoint_path, strict=False)

        img_unet_class = load_module(self.cfg.model.img_unet_class)
        pretrain_unet = UNet2DConditionModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="unet")
        img_unet_param = OmegaConf.to_container(self.cfg.model.img_unet, resolve=True)
        img_unet = img_unet_class.from_unet_2d_condition(pretrain_unet, **img_unet_param)

        unet_class = load_module(self.cfg.model.unet_class)
        self.unet = unet_class(img_unet=img_unet, pc_unet=pc_unet, **cfg.model.unet)


    def _calculate_steps(self):
        if self.train_dataloader is None:
            return  # there is no train dataloader, no need to set anything

        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.cfg.runner.gradient_accumulation_steps
        )
        # here the logic takes steps as higher priority. once set, will override
        # epochs param
        if self.overrode_max_train_steps:
            self.cfg.runner.max_train_steps = (
                self.cfg.runner.num_train_epochs * self.num_update_steps_per_epoch
            )
        else:
            # make sure steps and epochs are consistent
            self.cfg.runner.num_train_epochs = math.ceil(
                self.cfg.runner.max_train_steps / self.num_update_steps_per_epoch
            )

    def set_ema_models(self):
        self.models = [self.unet, self.pc_bbox_embedder, self.img_bbox_embedder]
        if self.use_ema:
            self.ema_models = [EMAModel(model.parameters()) for model in self.models]

            if self.cfg.resume_from_checkpoint:
                ema_unet_ckpt_dict = torch.load(os.path.join(self.cfg.resume_from_checkpoint, "ema_model_0.pth"), map_location="cpu")
                self.ema_models[0].load_state_dict(ema_unet_ckpt_dict)

                pc_box_embedder_ckpt_dict = torch.load(os.path.join(self.cfg.resume_from_checkpoint, "ema_model_1.pth"), map_location="cpu")
                self.ema_models[1].load_state_dict(pc_box_embedder_ckpt_dict)

                img_box_embedder_ckpt_dict = torch.load(os.path.join(self.cfg.resume_from_checkpoint, "ema_model_2.pth"), map_location="cpu")
                self.ema_models[2].load_state_dict(img_box_embedder_ckpt_dict)
            
                for ema_model in self.ema_models:
                    ema_model.to(self.unet.device)


    def _set_dataset_loader(self):
        # dataset
        collate_fn_param = {
            "tokenizer": self.tokenizer,
            "template": self.cfg.dataset.template,
            "bbox_mode": self.cfg.model.bbox_mode,
            "bbox_view_shared": self.cfg.model.bbox_view_shared,
            "bbox_drop_ratio": self.cfg.runner.bbox_drop_ratio,
            "bbox_add_ratio": self.cfg.runner.bbox_add_ratio,
            "bbox_add_num": self.cfg.runner.bbox_add_num,
        }

        if self.train_dataset is not None:
            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset, shuffle=True,
                collate_fn=partial(
                    multi_bbox_collate_fn, is_train=True, **collate_fn_param),
                batch_size=self.cfg.runner.train_batch_size,
                num_workers=self.cfg.runner.num_workers, pin_memory=True,
                prefetch_factor=self.cfg.runner.prefetch_factor,
                persistent_workers=True,
            )
        if self.val_dataset is not None:
            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset, shuffle=False,
                collate_fn=partial(
                    multi_bbox_collate_fn, is_train=False, **collate_fn_param),
                batch_size=self.cfg.runner.validation_batch_size,
                num_workers=self.cfg.runner.num_workers,
                prefetch_factor=self.cfg.runner.prefetch_factor,
            )

    def _set_model_trainable_state(self, train=True):
        # set trainable status
        self.pc_vae.requires_grad_(False)
        self.img_vae.requires_grad_(False)

        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.pc_bbox_embedder.train(train)
        self.img_bbox_embedder.train(train)
        for name, mod in self.unet.trainable_module.items():
            logging.debug(
                f"[MultiLDMBoxRunner] set {name} to requires_grad = True")
            mod.requires_grad_(train)

    def _set_xformer_state(self):
        # xformer
        if self.cfg.runner.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logging.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly")

    # def _set_gradient_checkpointing(self):
    #     if hasattr(self.cfg.runner.enable_unet_checkpointing, "__len__"):
    #         self.unet.enable_gradient_checkpointing(
    #             self.cfg.runner.enable_unet_checkpointing)
    #     elif self.cfg.runner.enable_unet_checkpointing:
    #         self.unet.enable_gradient_checkpointing()

    def set_optimizer_scheduler(self):
        # optimizer and lr_schedulers
        if self.cfg.runner.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # Optimizer creation
        unet_params = self.unet.trainable_parameters
        params_to_optimize = unet_params + list(self.pc_bbox_embedder.parameters()) + list(self.img_bbox_embedder.parameters())
        self.optimizer = optimizer_class(
            params_to_optimize,
            lr=self.cfg.runner.learning_rate,
            betas=(self.cfg.runner.adam_beta1, self.cfg.runner.adam_beta2),
            weight_decay=self.cfg.runner.adam_weight_decay,
            eps=self.cfg.runner.adam_epsilon,
        )

        # lr scheduler
        self._calculate_steps()
        # fmt: off
        self.lr_scheduler = get_scheduler(
            self.cfg.runner.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.runner.lr_warmup_steps * self.cfg.runner.gradient_accumulation_steps,
            num_training_steps=self.cfg.runner.max_train_steps * self.cfg.runner.gradient_accumulation_steps,
            num_cycles=self.cfg.runner.lr_num_cycles,
            power=self.cfg.runner.lr_power,
        )
        # fmt: on

    def prepare_device(self):
        # accelerator
        (
            self.unet,
            self.pc_bbox_embedder,
            self.img_bbox_embedder,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.unet, self.pc_bbox_embedder, self.img_bbox_embedder, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move vae, unet and text_encoder to device and cast to weight_dtype
        self.pc_vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.img_vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)

        if self.cfg.runner.unet_in_fp16 and self.weight_dtype == torch.float16:
            self.pc_bbox_embedder.to(self.accelerator.device, dtype=self.weight_dtype)
            self.img_bbox_embedder.to(self.accelerator.device, dtype=self.weight_dtype)
            self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
            # move optimized params to fp32. TODO: is this necessary?
            if self.cfg.model.use_fp32_for_unet_trainable:
                # mod = self.unet.module
                mod = self.unet

                mod.to(dtype=torch.float32)
                mod._original_forward = mod.forward
                # autocast intermediate is necessary since others are fp16
                mod.forward = torch.cuda.amp.autocast(
                    dtype=torch.float16)(mod.forward)
                # we ensure output is always fp16
                mod.forward = convert_outputs_to_fp16(mod.forward)

                # mod = self.bbox_embedder.module
                mod = self.pc_bbox_embedder

                mod.to(dtype=torch.float32)
                mod._original_forward = mod.forward
                # autocast intermediate is necessary since others are fp16
                mod.forward = torch.cuda.amp.autocast(
                    dtype=torch.float16)(mod.forward)
                # we ensure output is always fp16
                mod.forward = convert_outputs_to_fp16(mod.forward)

                # mod = self.bbox_embedder.module
                mod = self.img_bbox_embedder

                mod.to(dtype=torch.float32)
                mod._original_forward = mod.forward
                # autocast intermediate is necessary since others are fp16
                mod.forward = torch.cuda.amp.autocast(
                    dtype=torch.float16)(mod.forward)
                # we ensure output is always fp16
                mod.forward = convert_outputs_to_fp16(mod.forward)

            else:
                raise TypeError(
                    "There is an error/bug in accumulation wrapper, please "
                    "make all trainable param in fp32.")
        unet = self.accelerator.unwrap_model(self.unet)
        pc_bbox_embedder = self.accelerator.unwrap_model(self.pc_bbox_embedder)
        img_bbox_embedder = self.accelerator.unwrap_model(self.img_bbox_embedder)
        unet.weight_dtype = self.weight_dtype
        unet.unet_in_fp16 = self.cfg.runner.unet_in_fp16

        # We need to recalculate our total training steps as the size of the
        # training dataloader may have changed.
        self._calculate_steps()

        if self.use_ema:
            self.set_ema_models()
            # self.accelerator.register_save_state_pre_hook(self.save_model_hook)
            # self.accelerator.register_load_state_pre_hook(self.load_model_hook)


    def run(self):
        # Train!
        total_batch_size = (
            self.cfg.runner.train_batch_size
            * self.accelerator.num_processes
            * self.cfg.runner.gradient_accumulation_steps
        )

        # fmt: off
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(self.train_dataset)}")
        logging.info(f"  Num batches each epoch = {len(self.train_dataloader)}")
        logging.info(f"  Num Epochs = {self.cfg.runner.num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {self.cfg.runner.train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {self.cfg.runner.gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {self.cfg.runner.max_train_steps}")
        # fmt: on
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.cfg.resume_from_checkpoint:
            if self.cfg.resume_from_checkpoint != "latest":
                path = os.path.basename(self.cfg.resume_from_checkpoint)
            else:
                raise RuntimeError("We do not support in-place resume.")
                # Get the most recent checkpoint
                dirs = os.listdir(self.cfg.log_root)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.cfg.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                self.accelerator.print(
                    f"Resuming from checkpoint {self.cfg.resume_from_checkpoint}"
                )
                load_path = prepare_ckpt(
                    self.cfg.resume_from_checkpoint,
                    self.accelerator.is_local_main_process
                )
                self.accelerator.wait_for_everyone()  # wait
                if self.cfg.resume_reset_scheduler:
                    # reset to prevent from loading
                    self.accelerator._schedulers = []
                if self.cfg.resume_reset_optimizer:
                    self.accelerator._optimizers = []
                # load resume
                self.accelerator.load_state(load_path)
                global_step = int(path.split("-")[1])
                if self.cfg.resume_reset_scheduler:
                    # now we load some parameters for scheduler
                    resume_all_scheduler(self.lr_scheduler, load_path)
                    self.accelerator._schedulers = [self.lr_scheduler]
                if self.cfg.resume_reset_optimizer:
                    self.accelerator._optimizers = [self.optimizer]
                initial_global_step = (
                    global_step * self.cfg.runner.gradient_accumulation_steps
                )
                first_epoch = global_step // self.num_update_steps_per_epoch

        else:
            initial_global_step = 0

        # val before train
        if self.cfg.runner.validation_before_run or self.cfg.validation_only:
            if self.accelerator.is_main_process:
                self._validation(global_step)
            self.accelerator.wait_for_everyone()
            # if validation_only, exit
            if self.cfg.validation_only:
                self.accelerator.end_training()
                return

        # start train
        progress_bar = tqdm(
            range(0, self.cfg.runner.max_train_steps),
            initial=initial_global_step // self.cfg.runner.gradient_accumulation_steps,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
            miniters=self.num_update_steps_per_epoch // self.cfg.runner.display_per_epoch,
            maxinterval=self.cfg.runner.display_per_n_min * 60,
        )
        image_logs = None
        logging.info(
            f"Starting from epoch {first_epoch} to {self.cfg.runner.num_train_epochs}")
        for epoch in range(first_epoch, self.cfg.runner.num_train_epochs):
            for step, batch in enumerate(self.train_dataloader):
                loss, loss_pc, loss_img = self._train_one_stop(batch)
                if not loss.isfinite():
                    raise RuntimeError('Your loss is NaN.')
                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    # validation
                    if self.accelerator.is_main_process:
                        if global_step % self.cfg.runner.validation_steps == 0:
                            self._validation(global_step)
                    self.accelerator.wait_for_everyone()
                    # save and transfer
                    if global_step % self.cfg.runner.checkpointing_steps == 0:
                        sub_dir_name = f"checkpoint-{global_step}"
                        save_path = os.path.join(
                            self.cfg.log_root, sub_dir_name
                        )
                        self.accelerator.save_state(save_path)
                        if self.accelerator.is_main_process:
                            for e_id, ema_model in enumerate(self.ema_models):
                                torch.save(ema_model.state_dict(), os.path.join(save_path, "ema_model_%d.pth"%e_id))
                        logging.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "loss_pc": loss_pc.detach().item(), "loss_img": loss_img.detach().item()}
                for lri, lr in enumerate(self.lr_scheduler.get_last_lr()):
                    logs[f"lr{lri}"] = lr
                progress_bar.set_postfix(refresh=False, **logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= self.cfg.runner.max_train_steps:
                    break
            else:
                # on epoch end
                if self.cfg.runner.save_model_per_epoch is not None:
                    if epoch % self.cfg.runner.save_model_per_epoch == 0:
                        logging.info(
                            f"Save at step {global_step}, epoch {epoch}")
                        self.accelerator.wait_for_everyone()
                        sub_dir_name = f"weight-E{epoch}-S{global_step}"
                        self._save_model(os.path.join(
                            self.cfg.log_root, sub_dir_name
                        ))
                self.accelerator.wait_for_everyone()
                continue  # rather than break
            break  # if inner loop break, break again
        self.accelerator.wait_for_everyone()
        self._save_model()
        self.accelerator.end_training()

    def _save_model(self, root=None):
        if root is None:
            root = self.cfg.log_root
        # if self.accelerator.is_main_process:
        pc_bbox_embedder = self.accelerator.unwrap_model(self.pc_bbox_embedder)
        pc_bbox_embedder.save_pretrained(os.path.join(root, self.cfg.model.unet_dir))
        img_bbox_embedder = self.accelerator.unwrap_model(self.img_bbox_embedder)
        img_bbox_embedder.save_pretrained(os.path.join(root, self.cfg.model.unet_dir))
        unet = self.accelerator.unwrap_model(self.unet)
        unet.save_pretrained(os.path.join(root, self.cfg.model.unet_dir))
        logging.info(f"Save your model to: {root}")


    def _add_noise(self, latents, noise, timesteps, noise_scheduler):
        if self.cfg.runner.noise_offset > 0.0:
            # noise offset in SDXL, see:
            # https://github.com/Stability-AI/generative-models/blob/45c443b316737a4ab6e40413d7794a7f5657c19f/sgm/modules/diffusionmodules/loss.py#L47
            # they dit not apply on different channels. Don't know why.
            offset = self.cfg.runner.noise_offset * append_dims(
                torch.randn(latents.shape[:2], device=latents.device),
                latents.ndim
            ).type_as(latents)
            if self.cfg.runner.train_with_same_offset:
                offset = offset[:, :1]
            noise = noise + offset
        if timesteps.ndim == 2:
            B, N = latents.shape[:2]
            bc2b = partial(rearrange, pattern="b n ... -> (b n) ...")
            b2bc = partial(rearrange, pattern="(b n) ... -> b n ...", b=B)
        elif timesteps.ndim == 1:
            def bc2b(x): return x
            def b2bc(x): return x
        noisy_latents = noise_scheduler.add_noise(
            bc2b(latents), bc2b(noise), bc2b(timesteps)
        )
        noisy_latents = b2bc(noisy_latents)
        return noisy_latents

    def _train_one_stop(self, batch):
        self.unet.train()
        self.pc_bbox_embedder.train()
        self.img_bbox_embedder.train()

        with self.accelerator.accumulate(self.unet):
            
            # Convert images to latent space
            range_img = batch['range_img'].to(dtype=self.weight_dtype)
            range_img = range_img.permute(0, 1, 3, 2)
            latents_pc = self.pc_vae.encode(range_img).latent_dist.sample()
            latents_pc = latents_pc * self.pc_vae.config.scaling_factor

            N_cam = batch["pixel_values"].shape[1]
            # Convert images to latent space
            latents_img = self.img_vae.encode(
                rearrange(batch["pixel_values"], "b n c h w -> (b n) c h w").to(
                    dtype=self.weight_dtype
                )
            ).latent_dist.sample()
            latents_img = latents_img * self.img_vae.config.scaling_factor
            # embed camera params, in (B, 6, 3, 7), out (B, 6, 189)

            # Sample noise that we'll add to the latents
            noise_pc = torch.randn_like(latents_pc)
            # make sure we use same noise for different views, only take the
            # first

            noise_img = torch.randn_like(latents_img)
            if self.cfg.model.train_with_same_noise:
                noise_img = repeat(noise_img[:, 0], "b ... -> b r ...", r=N_cam)


            bsz = latents_pc.shape[0]
            # Sample a random timestep for each image
            timesteps_pc = torch.randint(
                0,
                self.noise_scheduler_pc.config.num_train_timesteps,
                (bsz,),
                device=latents_pc.device,
            )
            timesteps_pc = timesteps_pc.long()

            if self.cfg.runner.same_noise:
                timesteps_img = repeat(timesteps_pc, "b -> (b n)", n=N_cam)
            else:
                timesteps_img = torch.randint(
                    0,
                    self.noise_scheduler_img.config.num_train_timesteps,
                    (bsz,),
                    device=latents_img.device,
                )
                timesteps_img = repeat(timesteps_img, "b -> (b n)", n=N_cam)

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)

            noisy_latents_pc = self._add_noise(latents_pc, noise_pc, timesteps_pc, self.noise_scheduler_pc)
            noisy_latents_img = self._add_noise(latents_img, noise_img, timesteps_img, self.noise_scheduler_img)

            pos_encoding = torch.zeros_like(noisy_latents_pc[:, :1])
            pos_encoding[:, :, 0, :] = 1
            noisy_latents_input_pc = torch.cat([noisy_latents_pc, pos_encoding], dim=1)

            # Get the text embedding for conditioning
            encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
            encoder_hidden_states_uncond = self.text_encoder(batch["uncond_ids"])[0]
            encoder_hidden_states_uncond = encoder_hidden_states_uncond.repeat(bsz, 1, 1)

            uncond_mask = torch.zeros([bsz, 1, 1], dtype=torch.bool).to(encoder_hidden_states.device)
            # uncond_num = int(bsz * self.cfg.model.drop_cond_ratio)
            # uncond_ids = random.sample(range(bsz), uncond_num)
            mask_random = np.random.random(bsz)
            uncond_ids = np.where(mask_random<self.cfg.runner.drop_cond_ratio)[0]
            uncond_mask[uncond_ids] = True
            encoder_hidden_states = torch.where(uncond_mask, encoder_hidden_states_uncond, encoder_hidden_states)

            text_attention_mask = batch["input_ids"] != self.tokenizer.pad_token_id
            # text_attention_mask[uncond_ids] = False
            # text_attention_mask[uncond_ids][:, 0] = True
            text_attention_mask[uncond_ids] = True

            bboxes_3d_data_pc = batch['kwargs']['bboxes_3d_data_pc']
            bboxes_3d_data_img = batch['kwargs']['bboxes_3d_data_img']

            if (bboxes_3d_data_pc is not None) and (bboxes_3d_data_img is not None):
                bbox_embedder_kwargs_pc = {}
                for k, v in bboxes_3d_data_pc.items():
                    bbox_embedder_kwargs_pc[k] = v.clone()

                b_box_pc, n_box_pc = bbox_embedder_kwargs_pc["bboxes"].shape[:2]
                for k in bboxes_3d_data_pc.keys():
                    bbox_embedder_kwargs_pc[k] = rearrange(
                        bbox_embedder_kwargs_pc[k], 'b n ... -> (b n) ...')

                bbox_embedder_kwargs_img = {}
                for k, v in bboxes_3d_data_img.items():
                    bbox_embedder_kwargs_img[k] = v.clone()

                b_box_pc, n_box_pc = bbox_embedder_kwargs_img["bboxes"].shape[:2]
                for k in bboxes_3d_data_img.keys():
                    bbox_embedder_kwargs_img[k] = rearrange(
                        bbox_embedder_kwargs_img[k], 'b n ... -> (b n) ...')
            else:
                bbox_embedder_kwargs_pc = {
                    'bboxes': torch.zeros([bsz, 1, 8, 3]).to(latents_pc.device),
                    'classes': torch.zeros([bsz, 1]).to(latents_pc.device).long(),
                    'masks': torch.zeros([bsz, 1]).to(latents_pc.device).bool(),
                }
                bbox_embedder_kwargs_img = {
                    'bboxes': torch.zeros([bsz*N_cam, 1, 8, 3]).to(latents_pc.device),
                    'classes': torch.zeros([bsz*N_cam, 1]).to(latents_pc.device).long(),
                    'masks': torch.zeros([bsz*N_cam, 1]).to(latents_pc.device).bool(),
                }
        
            lidar2imgs = batch["lidar2imgs"].to(self.weight_dtype)
            bbox_embedder_kwargs_img['lidar2imgs'] = lidar2imgs

            bbox_emb_pc = self.pc_bbox_embedder(**bbox_embedder_kwargs_pc)
            # padding_mask = torch.logical_not(bbox_embedder_kwargs_img['masks'])
            bbox_emb_img, bbox_embedder_kwargs_img['masks'] = self.img_bbox_embedder(**bbox_embedder_kwargs_img)
            # bbox_embedder_kwargs_img['masks'] = torch.logical_or(bbox_embedder_kwargs_img['masks'], padding_mask)

            mask_random = np.random.random(bsz)
            box_uncond_ids = np.where(mask_random<self.cfg.runner.drop_cond_ratio)[0]

            box_uncond_mask = torch.zeros([bsz, 1], dtype=torch.bool).to(encoder_hidden_states.device)
            box_uncond_mask[box_uncond_ids] = True
            pc_box_num = bbox_emb_pc.shape[1]
            # pc_box_uncond_embed = self.pc_bbox_embedder.n_uncond_tokens(bbox_emb_pc, pc_box_num).type_as(bbox_emb_pc)
            if hasattr(self.pc_bbox_embedder, "module"):
                pc_box_uncond_embed = self.pc_bbox_embedder.module.n_uncond_tokens(bbox_emb_pc, pc_box_num).type_as(bbox_emb_pc)
            else:
                pc_box_uncond_embed = self.pc_bbox_embedder.n_uncond_tokens(bbox_emb_pc, pc_box_num).type_as(bbox_emb_pc)
            pc_box_uncond_mask = box_uncond_mask.repeat(1, pc_box_num)
            # uncond_bbox_embedder_pc_mask = torch.zeros_like(bbox_embedder_kwargs_pc['masks'])
            # uncond_bbox_embedder_pc_mask[:, 0] = True
            uncond_bbox_embedder_pc_mask = torch.ones_like(bbox_embedder_kwargs_pc['masks'])
            
            bbox_embedder_kwargs_pc['masks'] = torch.where(pc_box_uncond_mask, uncond_bbox_embedder_pc_mask, bbox_embedder_kwargs_pc['masks'])
            pc_box_uncond_mask = pc_box_uncond_mask[..., None].repeat(1, 1, pc_box_uncond_embed.shape[-1])
            bbox_emb_pc = torch.where(pc_box_uncond_mask, pc_box_uncond_embed, bbox_emb_pc)

            encoder_hidden_states_pc = [encoder_hidden_states.to(dtype=self.weight_dtype), bbox_emb_pc.to(dtype=self.weight_dtype), bbox_embedder_kwargs_pc['masks']]

            encoder_hidden_states_img = rearrange(repeat(encoder_hidden_states, "b ... -> b r ...", r=N_cam), 'b n ... -> (b n) ...')
            img_bbox_num = bbox_emb_img.shape[1]
            if hasattr(self.img_bbox_embedder, "module"):
                img_box_uncond_embed = self.img_bbox_embedder.module.n_uncond_tokens(bbox_emb_img, img_bbox_num).type_as(bbox_emb_img)
            else:
                img_box_uncond_embed = self.img_bbox_embedder.n_uncond_tokens(bbox_emb_img, img_bbox_num).type_as(bbox_emb_img)
            img_box_uncond_mask = box_uncond_mask[:,None].repeat(1, N_cam, img_bbox_num)
            img_box_uncond_mask = img_box_uncond_mask.flatten(0, 1)
            # uncond_bbox_embedder_img_mask = torch.zeros_like(bbox_embedder_kwargs_img['masks'])
            # uncond_bbox_embedder_img_mask[:, 0] = True
            uncond_bbox_embedder_img_mask = torch.ones_like(bbox_embedder_kwargs_img['masks'])

            bbox_embedder_kwargs_img['masks'] = torch.where(img_box_uncond_mask, uncond_bbox_embedder_img_mask, bbox_embedder_kwargs_img['masks'])
            img_box_uncond_mask = img_box_uncond_mask[..., None].repeat(1, 1, img_box_uncond_embed.shape[-1])
            bbox_emb_img = torch.where(img_box_uncond_mask, img_box_uncond_embed, bbox_emb_img)

            encoder_hidden_states_img = [encoder_hidden_states_img.to(dtype=self.weight_dtype), bbox_emb_img.to(dtype=self.weight_dtype), bbox_embedder_kwargs_img['masks']]

            text_attention_mask_img = text_attention_mask[:, None].repeat(1, N_cam, 1).flatten(0, 1)

            model_pred_pc, model_pred_img = self.unet(
                noisy_latents_input_pc,
                noisy_latents_img,

                timesteps_pc.reshape(-1), 
                timesteps_img.reshape(-1), 

                encoder_hidden_states_pc=encoder_hidden_states_pc,  # b x n, len + 1, 768
                encoder_hidden_states_img=encoder_hidden_states_img,  # b x n, len + 1, 768
                attention_mask_pc=text_attention_mask,
                attention_mask_img=text_attention_mask_img,

                lidar2imgs=lidar2imgs

                # TODO: during training, some camera param are masked.
            )
            model_pred_pc = model_pred_pc.sample
            model_pred_img = model_pred_img.sample

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler_pc.config.prediction_type == "epsilon":
                target_pc = noise_pc
            elif self.noise_scheduler_pc.config.prediction_type == "v_prediction":
                target_pc = self.noise_scheduler_pc.get_velocity(latents_pc, noise_pc, timesteps_pc)
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler_pc.config.prediction_type}"
                )

            if self.noise_scheduler_img.config.prediction_type == "epsilon":
                target_img = noise_img
            elif self.noise_scheduler_img.config.prediction_type == "v_prediction":
                target_img = self.noise_scheduler_img.get_velocity(latents_img, noise_img, timesteps_img)
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler_img.config.prediction_type}"
                )

            loss_pc = F.mse_loss(model_pred_pc.float(), target_pc.float(), reduction='none')
            loss_pc = loss_pc.mean()
            loss_img = F.mse_loss(model_pred_img.float(), target_img.float(), reduction='none')
            loss_img = loss_img.mean()

            loss = (loss_pc + loss_img) / 2

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients: 
                params_to_clip = list(self.unet.parameters()) + list(self.pc_bbox_embedder.parameters()) + list(self.img_bbox_embedder.parameters())
                self.accelerator.clip_grad_norm_(
                    params_to_clip, self.cfg.runner.max_grad_norm
                )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(
                set_to_none=self.cfg.runner.set_grads_to_none)

            if self.use_ema:
                for model, ema_model in zip(self.models, self.ema_models):
                    model = self.accelerator.unwrap_model(model)
                    ema_model.step(model.parameters())

        return loss, loss_pc, loss_img

    def _validation(self, step):
        bbox_embedder_pc = self.accelerator.unwrap_model(self.pc_bbox_embedder)
        bbox_embedder_img = self.accelerator.unwrap_model(self.img_bbox_embedder)

        unet = self.accelerator.unwrap_model(self.unet)
        image_logs = self.validator.validate(
            bbox_embedder_pc, bbox_embedder_img, unet, self.accelerator.trackers, step,
            self.weight_dtype, self.accelerator.device)
