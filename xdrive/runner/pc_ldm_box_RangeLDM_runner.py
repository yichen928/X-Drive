import os
import math
import logging
from omegaconf import OmegaConf
from functools import partial
from packaging import version
from tqdm.auto import tqdm

import random
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from diffusers.training_utils import EMAModel
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
    UNet2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from accelerate import Accelerator
import safetensors

from xdrive.dataset.utils import lidar_bbox_collate_fn
from xdrive.runner.pc_validator_box import PCValidatorBox
from xdrive.runner.utils import (
    prepare_ckpt,
    resume_all_scheduler,
    append_dims,
)
from xdrive.misc.common import (
    move_to,
    load_module,
    deepspeed_zero_init_disabled_context_manager,
)

from ..misc.common import load_module, convert_outputs_to_fp16, move_to
from .base_runner import BaseRunner

from third_party.RangeLDM.ldm.utils import replace_attn, replace_conv, replace_down
from xdrive.networks.unet_pc_condition_RangeLDM import RangeLDMPCUNet2DModel
torch.manual_seed(42)

class RangeLDMPCBoxRunner(BaseRunner):
    def __init__(self, cfg, accelerator, train_set, val_set) -> None:
        super().__init__(cfg, accelerator, train_set, val_set)

        self.use_ema = self.cfg.runner.use_ema
        pipe_cls = load_module(self.cfg.model.pipe_module)
        self.validator = PCValidatorBox(
            self.cfg,
            self.val_dataset,
            pipe_cls,
            pipe_param={
                "vae": self.vae,
                "text_encoder": self.text_encoder,
                "tokenizer": self.tokenizer,
                "scheduler": self.noise_scheduler
            }
        )

    def _init_fixed_models(self, cfg):
        # fmt: off
        torch.manual_seed(42)
        
        print("Pretrained model path:", cfg.model.pretrained_model_name_or_path)

        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(cfg.model.pretrained_RangeLDM_model_name_or_path, subfolder="scheduler")

        vae_config_path = os.path.join(cfg.model.pretrained_RangeLDM_model_name_or_path, "vae", "config.json")
        vae_checkpoint_path = os.path.join(cfg.model.pretrained_RangeLDM_model_name_or_path, "vae", "diffusion_pytorch_model.safetensors")

        vae_config = AutoencoderKL.load_config(vae_config_path)
        vae = AutoencoderKL.from_config(vae_config)
        vae_checkpoint = safetensors.torch.load_file(vae_checkpoint_path)
        if 'quant_conv.weight' not in vae_checkpoint:
            vae.quant_conv = torch.nn.Identity()
            vae.post_quant_conv = torch.nn.Identity()
        replace_down(vae)
        replace_conv(vae)
        if 'encoder.mid_block.attentions.0.to_q.weight' not in vae_checkpoint:
            replace_attn(vae)
        vae.load_state_dict(vae_checkpoint)

        self.vae = vae
        # fmt: on

    def _init_trainable_models(self, cfg):
        unet_checkpoint_path = os.path.join(cfg.model.pretrained_RangeLDM_model_name_or_path, "unet", "diffusion_pytorch_model.safetensors")
        unet_config_path = os.path.join(cfg.model.pretrained_RangeLDM_model_name_or_path, "unet", "config.json")

        unet_config = RangeLDMPCUNet2DModel.load_config(unet_config_path)
        unet_config['down_block_types'] = ['DownBlock2D', 'CondAttnDownBlock2D', 'CondAttnDownBlock2D', 'CondAttnDownBlock2D']
        unet_config['up_block_types'] = ['CondAttnUpBlock2D', 'CondAttnUpBlock2D', 'CondAttnUpBlock2D', 'UpBlock2D']
        unet_config['cross_attention_dim'] = cfg.model.unet.cross_attention_dim
        unet_config['use_gsa'] = cfg.model.unet.use_gsa

        unet = RangeLDMPCUNet2DModel.from_config(unet_config)

        # unet_config = UNet2DModel.load_config(unet_config_path)
        # unet = UNet2DModel.from_config(unet_config)

        replace_down(unet)
        replace_conv(unet)
        safetensors.torch.load_model(unet, unet_checkpoint_path, strict=False)

        self.unet = unet

        model_cls = load_module(self.cfg.model.bbox_embedder_cls)
        self.bbox_embedder = model_cls(**self.cfg.model.bbox_embedder_param)
        self.bbox_embedder.prepare(self.cfg, **{"tokenizer": self.tokenizer, "text_encoder": self.text_encoder})


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
        self.models = [self.unet, self.bbox_embedder]
        if self.use_ema:
            self.ema_models = [EMAModel(model.parameters()) for model in self.models]

            if self.cfg.resume_from_checkpoint:
                ema_unet_ckpt_dict = torch.load(os.path.join(self.cfg.resume_from_checkpoint, "ema_model_0.pth"), map_location="cpu")
                self.ema_models[0].load_state_dict(ema_unet_ckpt_dict)

                pc_box_embedder_ckpt_dict = torch.load(os.path.join(self.cfg.resume_from_checkpoint, "ema_model_1.pth"), map_location="cpu")
                self.ema_models[1].load_state_dict(pc_box_embedder_ckpt_dict)

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
                    lidar_bbox_collate_fn, is_train=True, **collate_fn_param),
                batch_size=self.cfg.runner.train_batch_size,
                num_workers=self.cfg.runner.num_workers, pin_memory=True,
                prefetch_factor=self.cfg.runner.prefetch_factor,
                persistent_workers=True,
            )
        if self.val_dataset is not None:
            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset, shuffle=False,
                collate_fn=partial(
                    lidar_bbox_collate_fn, is_train=False, **collate_fn_param),
                batch_size=self.cfg.runner.validation_batch_size,
                num_workers=self.cfg.runner.num_workers,
                prefetch_factor=self.cfg.runner.prefetch_factor,
            )

    def _set_model_trainable_state(self, train=True):
        # set trainable status
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.train(train)
        self.bbox_embedder.train(train)

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

    def _set_gradient_checkpointing(self):
        if hasattr(self.cfg.runner.enable_unet_checkpointing, "__len__"):
            self.unet.enable_gradient_checkpointing(
                self.cfg.runner.enable_unet_checkpointing)
        elif self.cfg.runner.enable_unet_checkpointing:
            self.unet.enable_gradient_checkpointing()

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
        params_to_optimize = list(self.unet.parameters()) + list(self.bbox_embedder.parameters())
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
        # for name, param in self.unet.named_parameters():
        #     if "pos" in name:
        #         print(f"{name}: {param.size()}", end="   ")
        
        # print()

        (
            self.unet,
            self.bbox_embedder,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.unet, self.bbox_embedder, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move vae, unet and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)

        if self.cfg.runner.unet_in_fp16 and self.weight_dtype == torch.float16:
            self.bbox_embedder.to(self.accelerator.device, dtype=self.weight_dtype)
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
                mod = self.bbox_embedder

                mod.to(dtype=torch.float32)
                mod._original_forward = mod.forward
                # autocast intermediate is necessary since others are fp16
                mod.forward = torch.cuda.amp.autocast(
                    dtype=torch.float16)(mod.forward)
                # we ensure output is always fp16
                mod.forward = convert_outputs_to_fp16(mod.forward)
                # for name, mod in self.unet.items():
                #     logging.debug(f"[PCLDMRunner] set {name} to fp32")
                #     mod.to(dtype=torch.float32)
                #     mod._original_forward = mod.forward
                #     # autocast intermediate is necessary since others are fp16
                #     mod.forward = torch.cuda.amp.autocast(
                #         dtype=torch.float16)(mod.forward)
                #     # we ensure output is always fp16
                #     mod.forward = convert_outputs_to_fp16(mod.forward)
                # for name, mod in self.bbox_embedder.items():
                #     logging.debug(f"[PCLDMRunner] set {name} to fp32")
                #     mod.to(dtype=torch.float32)
                #     mod._original_forward = mod.forward
                #     # autocast intermediate is necessary since others are fp16
                #     mod.forward = torch.cuda.amp.autocast(
                #         dtype=torch.float16)(mod.forward)
                #     # we ensure output is always fp16
                #     mod.forward = convert_outputs_to_fp16(mod.forward)
            else:
                raise TypeError(
                    "There is an error/bug in accumulation wrapper, please "
                    "make all trainable param in fp32.")
        unet = self.accelerator.unwrap_model(self.unet)
        bbox_embedder = self.accelerator.unwrap_model(self.bbox_embedder)
        unet.weight_dtype = self.weight_dtype
        unet.unet_in_fp16 = self.cfg.runner.unet_in_fp16

        # We need to recalculate our total training steps as the size of the
        # training dataloader may have changed.
        self._calculate_steps()

        if self.use_ema:
            self.set_ema_models()

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
                # load resume
                self.accelerator.load_state(load_path)
                global_step = int(path.split("-")[1])
                if self.cfg.resume_reset_scheduler:
                    # now we load some parameters for scheduler
                    resume_all_scheduler(self.lr_scheduler, load_path)
                    self.accelerator._schedulers = [self.lr_scheduler]
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
                loss = self._train_one_stop(batch)
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

                logs = {"loss": loss.detach().item()}
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
        bbox_embedder = self.accelerator.unwrap_model(self.bbox_embedder)
        bbox_embedder.save_pretrained(os.path.join(root, self.cfg.model.unet_dir))
        unet = self.accelerator.unwrap_model(self.unet)
        unet.save_pretrained(os.path.join(root, self.cfg.model.unet_dir))
        logging.info(f"Save your model to: {root}")


    def _add_noise(self, latents, noise, timesteps):
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
        noisy_latents = self.noise_scheduler.add_noise(
            bc2b(latents), bc2b(noise), bc2b(timesteps)
        )
        noisy_latents = b2bc(noisy_latents)
        return noisy_latents

    def _train_one_stop(self, batch):
        self.unet.train()
        self.bbox_embedder.train()

        with self.accelerator.accumulate(self.unet):

            # Convert images to latent space
            range_img = batch['range_img'].to(dtype=self.weight_dtype)
            range_img = range_img.permute(0, 1, 3, 2)
            with torch.no_grad():
                latents = self.vae.encode(range_img).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            # embed camera params, in (B, 6, 3, 7), out (B, 6, 189)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            # make sure we use same noise for different views, only take the
            # first
            if self.cfg.model.train_with_same_noise:
                noise = repeat(noise[:, 0], "b ... -> b r ...", r=N_cam)

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self._add_noise(latents, noise, timesteps)
            
            # Get the text embedding for conditioning
            encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
            encoder_hidden_states_uncond = self.text_encoder(batch["uncond_ids"])[0]
            encoder_hidden_states_uncond = encoder_hidden_states_uncond.repeat(bsz, 1, 1)

            uncond_mask = torch.zeros([bsz, 1, 1], dtype=torch.bool).to(encoder_hidden_states.device)
            uncond_num = int(bsz * self.cfg.model.drop_cond_ratio)
            uncond_ids = random.sample(range(bsz), uncond_num)
            uncond_mask[uncond_ids] = True
            encoder_hidden_states = torch.where(uncond_mask, encoder_hidden_states_uncond, encoder_hidden_states)

            text_attention_mask = batch["input_ids"] != self.tokenizer.pad_token_id
            text_attention_mask[uncond_ids] = True


            bboxes_3d_data = batch['kwargs']['bboxes_3d_data']
            bbox_embedder_kwargs = {}
            for k, v in bboxes_3d_data.items():
                bbox_embedder_kwargs[k] = v.clone()

            b_box, n_box = bbox_embedder_kwargs["bboxes"].shape[:2]
            for k in bboxes_3d_data.keys():
                bbox_embedder_kwargs[k] = rearrange(
                    bbox_embedder_kwargs[k], 'b n ... -> (b n) ...')

            bbox_emb = self.bbox_embedder(**bbox_embedder_kwargs)

            box_uncond_ids = random.sample(range(bsz), uncond_num)
            box_uncond_mask = torch.zeros([bsz, 1], dtype=torch.bool).to(encoder_hidden_states.device)
            box_uncond_mask[box_uncond_ids] = True
            box_num = bbox_emb.shape[1]
            box_uncond_embed = self.bbox_embedder.module.n_uncond_tokens(bbox_emb, box_num).type_as(bbox_emb)
            box_uncond_mask = box_uncond_mask.repeat(1, box_num)
            bbox_embedder_kwargs['masks'] = torch.where(box_uncond_mask, torch.ones_like(box_uncond_mask), bbox_embedder_kwargs['masks'])
            box_uncond_mask = box_uncond_mask[..., None].repeat(1, 1, box_uncond_embed.shape[-1])
            bbox_emb = torch.where(box_uncond_mask, box_uncond_embed, bbox_emb)

            encoder_hidden_states_mix = [encoder_hidden_states.to(dtype=self.weight_dtype), bbox_emb.to(dtype=self.weight_dtype), bbox_embedder_kwargs['masks']]
            
            pos_encoding = torch.zeros_like(noisy_latents[:, :1])
            pos_encoding[:, :, 0, :] = 1
            noisy_latent_model_input = torch.cat([noisy_latents, pos_encoding], dim=1)

            model_pred = self.unet(
                noisy_latent_model_input,  # b x n, 4, H/8, W/8
                timesteps.reshape(-1),  # b x n
                encoder_hidden_states=encoder_hidden_states_mix,  # b x n, len + 1, 768
                attention_mask=text_attention_mask,
                # TODO: during training, some camera param are masked.
            ).sample

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(
                    latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                )

            loss = F.mse_loss(
                model_pred.float(), target.float(), reduction='none')
            loss = loss.mean()

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                params_to_clip = list(self.unet.parameters()) + list(self.bbox_embedder.parameters())
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

        return loss

    def _validation(self, step):
        bbox_embedder = self.accelerator.unwrap_model(self.bbox_embedder)
        unet = self.accelerator.unwrap_model(self.unet)
        image_logs = self.validator.validate(
            bbox_embedder, unet, self.accelerator.trackers, step,
            self.weight_dtype, self.accelerator.device)
