from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import inspect

import torch
import PIL
import numpy as np
from einops import rearrange, repeat

from diffusers import StableDiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    is_compiled_module,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from ..misc.common import move_to


@dataclass
class BEVStableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_content_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, or `None` if safety checking could not be performed.
    """

    images: Union[List[List[PIL.Image.Image]], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]


class RangeLDMMultiBoxPipeline(
    DiffusionPipeline,
    TextualInversionLoaderMixin,
    LoraLoaderMixin,
):
    def __init__(
        self,
        vae_pc,
        vae_img,
        text_encoder: CLIPTextModel,
        unet: UNet2DConditionModel,
        bbox_embedder_pc,
        bbox_embedder_img,
        scheduler_pc: KarrasDiffusionSchedulers,
        scheduler_img: KarrasDiffusionSchedulers,
        tokenizer: CLIPTokenizer,
        safety_checker: StableDiffusionSafetyChecker = None,
        feature_extractor: CLIPImageProcessor = None,
        requires_safety_checker: bool = False,
        only_point_clouds: bool = False,
        only_images: bool = False,
        wo_boxes: bool = False,
        wo_texts: bool = False,
    ):
        super().__init__()

        if hasattr(scheduler_pc.config, "clip_sample") and scheduler_pc.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler_pc} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler_pc.config)
            new_config["clip_sample"] = False
            scheduler_pc._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler_img.config, "clip_sample") and scheduler_img.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler_img} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler_img.config)
            new_config["clip_sample"] = False
            scheduler_img._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        # is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        # if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
        #     deprecation_message = (
        #         "The configuration file of the unet has set the default `sample_size` to smaller than"
        #         " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
        #         " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
        #         " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
        #         " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
        #         " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
        #         " in the config might lead to incorrect results in future versions. If you have downloaded this"
        #         " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
        #         " the `unet/config.json` file"
        #     )
        #     deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
        #     new_config = dict(unet.config)
        #     new_config["sample_size"] = 64
        #     unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae_pc=vae_pc,
            vae_img=vae_img,
            bbox_embedder_pc=bbox_embedder_pc,
            bbox_embedder_img=bbox_embedder_img,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler_pc=scheduler_pc,
            scheduler_img=scheduler_img,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor_pc = 2 ** (len(self.vae_pc.config.block_out_channels) - 1)
        self.vae_scale_factor_img = 2 ** (len(self.vae_img.config.block_out_channels) - 1)
        self.pc_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor_pc)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor_img)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

        self.only_point_clouds = only_point_clouds
        self.only_images = only_images
        assert not (only_point_clouds and only_images)

        self.wo_texts = wo_texts
        self.wo_boxes = wo_boxes

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    def numpy_to_pil_double(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        We need to handle 5-dim inputs and reture 2-dim list.
        """
        imgs_list = []
        for imgs in images:
            imgs_list.append(self.numpy_to_pil(imgs))
        return imgs_list

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, scheduler, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        # accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
        # if accepts_generator:
        #     raise RuntimeError("If you fixed the logic for generator, please remove this. Otherwise, please use other sampler.")
        #     extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def decode_latents(self, latents, vae):
        # decode latents with 5-dims
        latents = 1 / vae.config.scaling_factor * latents

        bs = len(latents)
        image = vae.decode(latents).sample
        image = image.cpu().float().numpy()

        return image


    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, vae_scale_factor, scheduler, latents=None):
        shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler.init_noise_sigma
        return latents

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        camera_param: Union[torch.Tensor, None],
        range_height: int,
        range_width: int,
        height: int,
        width: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents_pc: Optional[torch.FloatTensor] = None,
        latents_img: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: float = 1,
        guess_mode: bool = False,
        use_zero_map_as_unconditional: bool = False,
        val_data_kwargs = {},
        bbox_max_length = None,
        lidar2imgs = None,

    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`,
                    `List[List[torch.FloatTensor]]`, or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
                the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can
                also be accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If
                height and/or width are passed, `image` is resized according to them. If multiple ControlNets are
                specified in init, images must be passed as a list such that each element of the list can be correctly
                batched for input to a single controlnet.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet. If multiple ControlNets are specified in init, you can set the
                corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                In this mode, the ControlNet encoder will try best to recognize the content of the input image even if
                you remove all prompts. The `guidance_scale` between 3.0 and 5.0 is recommended.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        # BEV: we cannot use the size of image
        # height, width = self._default_height_width(height, width, None)

        # 1. Check inputs. Raise error if not correct
        # we do not need this, only some type assertion
        # self.check_inputs(
        #     prompt,
        #     image,
        #     height,
        #     width,
        #     callback_steps,
        #     negative_prompt,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     controlnet_conditioning_scale,
        # )

        # 2. Define call parameters
        # NOTE: we get batch_size first from prompt, then align with it.
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        lidar2imgs = lidar2imgs.to(self.device)
        if do_classifier_free_guidance:
            lidar2imgs = torch.cat([lidar2imgs, lidar2imgs], dim=0)

        if camera_param is None:
            # use uncond_cam and disable classifier free guidance
            N_cam = 6  # TODO: hard-coded
            camera_param = self.controlnet.uncond_cam_param((batch_size, N_cam))
            do_classifier_free_guidance = False

        # 3. Encode input prompt
        # NOTE: here they use padding to 77, is this necessary?
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 5. Prepare timesteps
        if self.only_point_clouds:
            self.scheduler_pc.set_timesteps(num_inference_steps, device=device)
            self.scheduler_img.set_timesteps(0, device=device)
            timesteps = self.scheduler_pc.timesteps    
        elif self.only_images:
            self.scheduler_pc.set_timesteps(0, device=device)
            self.scheduler_img.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler_img.timesteps    
        else:
            self.scheduler_pc.set_timesteps(num_inference_steps, device=device)
            self.scheduler_img.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler_pc.timesteps

        N_cam = camera_param.shape[1]

        # 6. Prepare latent variables
        num_channels_latents_pc = self.unet.pc_unet.config.out_channels
        num_channels_latents_img = self.unet.img_unet.config.in_channels

        latents_pc = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents_pc,
            range_width,
            range_height, # RangeLDM requires width x height instead of height x width
            prompt_embeds.dtype,
            device,
            generator,
            self.vae_scale_factor_pc,
            self.scheduler_pc,
            latents_pc,  # will use if not None, otherwise will generate
        )  # (b, c, h/8, w/8) -> (bs, 4, 28, 50)

        latents_img = self.prepare_latents(
            batch_size * N_cam * num_images_per_prompt,
            num_channels_latents_img,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            self.vae_scale_factor_img,
            self.scheduler_img,
            latents_img,  # will use if not None, otherwise will generate
        )  # (b, c, h/8, w/8) -> (bs, 4, 28, 50)

        # 7. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, self.scheduler_pc, eta)

        bboxes_3d_data_pc = val_data_kwargs['bboxes_3d_data_pc']
        bboxes_3d_data_img = val_data_kwargs['bboxes_3d_data_img']
        
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
                'bboxes': torch.zeros([batch_size, 1, 8, 3]).to(device),
                'classes': torch.zeros([batch_size, 1]).to(device).long(),
                'masks': torch.zeros([batch_size, 1]).to(device).bool(),
            }
            bbox_embedder_kwargs_img = {
                'bboxes': torch.zeros([batch_size*N_cam, 1, 8, 3]).to(device),
                'classes': torch.zeros([batch_size*N_cam, 1]).to(device).long(),
                'masks': torch.zeros([batch_size*N_cam, 1]).to(device).bool(),
            }
            
        # bboxes_3d_data_pc = val_data_kwargs['bboxes_3d_data_pc']
        # bbox_embedder_kwargs_pc = {}
        # for k, v in bboxes_3d_data_pc.items():
        #     bbox_embedder_kwargs_pc[k] = v.clone()

        # b_box, n_box = bbox_embedder_kwargs_pc["bboxes"].shape[:2]
        # for k in bboxes_3d_data_pc.keys():
        #     bbox_embedder_kwargs_pc[k] = rearrange(
        #         bbox_embedder_kwargs_pc[k], 'b n ... -> (b n) ...')


        # bboxes_3d_data_img = val_data_kwargs['bboxes_3d_data_img']
        # bbox_embedder_kwargs_img = {}
        # for k, v in bboxes_3d_data_img.items():
        #     bbox_embedder_kwargs_img[k] = v.clone()

        # b_box, n_box = bbox_embedder_kwargs_img["bboxes"].shape[:2]
        # for k in bboxes_3d_data_img.keys():
        #     bbox_embedder_kwargs_img[k] = rearrange(
        #         bbox_embedder_kwargs_img[k], 'b n ... -> (b n) ...')

        bbox_embedder_kwargs_img['lidar2imgs'] = lidar2imgs[0:1]
        for key in bbox_embedder_kwargs_img:
            bbox_embedder_kwargs_img[key] = bbox_embedder_kwargs_img[key].to(device)

        bbox_emb_pc = self.bbox_embedder_pc(**bbox_embedder_kwargs_pc)
        bbox_emb_img, bbox_embedder_kwargs_img['masks'] = self.bbox_embedder_img(**bbox_embedder_kwargs_img)

        text_len = prompt_embeds.shape[1]
        # uncond_token = self.bbox_embedder.forward_feature(
        #     self.bbox_embedder.null_pos_feature[None], self.bbox_embedder.null_class_feature[None])
        # uncond_token = repeat(uncond_token[0], 'c -> b n c', b=negative_prompt_embeds.shape[0], n=bbox_emb.shape[1])
        # negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_token], dim=1)

        img_box_uncond_embed = self.bbox_embedder_img.n_uncond_tokens(bbox_emb_img, bbox_emb_img.shape[1]).type_as(bbox_emb_img)
        pc_box_uncond_embed = self.bbox_embedder_pc.n_uncond_tokens(bbox_emb_pc, bbox_emb_pc.shape[1]).type_as(bbox_emb_pc)

        negative_prompt_embeds_pc = torch.cat([negative_prompt_embeds, pc_box_uncond_embed], dim=1)
        negative_prompt_embeds_img = torch.cat([negative_prompt_embeds.expand(N_cam, -1, -1), img_box_uncond_embed], dim=1)

        if self.wo_boxes:
            bbox_emb_pc = pc_box_uncond_embed
            bbox_emb_img = img_box_uncond_embed
        if self.wo_texts:
            prompt_embeds = negative_prompt_embeds
            
        prompt_embeds_pc = torch.cat([prompt_embeds, bbox_emb_pc], dim=1)
        prompt_embeds_img = torch.cat([prompt_embeds.expand(N_cam, -1, -1), bbox_emb_img], dim=1)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler_pc.order
        latents_pc_0 = latents_pc.clone()
        latents_img_0 = latents_img.clone()
        noise_pc = torch.randn_like(latents_pc)
        noise_img = torch.randn_like(latents_img)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.only_point_clouds:
                    t_pc = t
                    t_img = 0
                elif self.only_images:
                    t_pc = 0
                    t_img = t
                else:
                    t_pc = t
                    t_img = t
                latents_pc_ = latents_pc
                latents_img_ = latents_img
                # t_pc = t
                # t_img = t
                # if self.only_images:
                #     latents_pc_ = self.scheduler_img.add_noise(latents_pc_0, noise_pc, t)
                # elif self.only_point_clouds:
                #     latents_img_ = self.scheduler_img.add_noise(latents_img_0, noise_img, t)

                # expand the latents if we are doing classifier free guidance
                latent_model_input_pc = (
                    torch.cat([latents_pc_] * 2) if do_classifier_free_guidance else latents_pc_
                )
                latent_model_input_pc = self.scheduler_pc.scale_model_input(
                    latent_model_input_pc, t_pc
                )

                pos_encoding = torch.zeros_like(latent_model_input_pc[:, :1])
                pos_encoding[:, :, 0, :] = 1

                latent_model_input_pc = torch.cat([latent_model_input_pc, pos_encoding], dim=1)

                latent_model_input_img = (
                    torch.cat([latents_img_] * 2) if do_classifier_free_guidance else latents_img_
                )
                latent_model_input_img = self.scheduler_img.scale_model_input(
                    latent_model_input_img, t_img
                )

                encoder_hidden_states_pc = torch.cat([negative_prompt_embeds_pc, prompt_embeds_pc], dim=0)

                # uncond_bbox_embedder_kwargs_pc_mask = torch.zeros_like(bbox_embedder_kwargs_pc['masks'])
                # uncond_bbox_embedder_kwargs_pc_mask[:, 0] = True
                uncond_bbox_embedder_kwargs_pc_mask = torch.ones_like(bbox_embedder_kwargs_pc['masks'])

                encoder_hidden_states_pc = [encoder_hidden_states_pc[:, :text_len], encoder_hidden_states_pc[:, text_len:], torch.cat([uncond_bbox_embedder_kwargs_pc_mask, bbox_embedder_kwargs_pc['masks']], dim=0).type_as(latent_model_input_pc)]

                encoder_hidden_states_img = torch.cat([negative_prompt_embeds_img, prompt_embeds_img], dim=0)
                # uncond_bbox_embedder_kwargs_img_mask = torch.zeros_like(bbox_embedder_kwargs_img['masks'])
                # uncond_bbox_embedder_kwargs_img_mask[:, 0] = True
                uncond_bbox_embedder_kwargs_img_mask = torch.ones_like(bbox_embedder_kwargs_img['masks'])

                encoder_hidden_states_img = [encoder_hidden_states_img[:, :text_len], encoder_hidden_states_img[:, text_len:], torch.cat([uncond_bbox_embedder_kwargs_img_mask, bbox_embedder_kwargs_img['masks']], dim=0).type_as(latent_model_input_img)]

                # uncond_text_attention_mask = torch.zeros([latents_pc_.shape[0], text_len]).type_as(encoder_hidden_states_pc[-1])
                # uncond_text_attention_mask[:, 0] = True
                uncond_text_attention_mask = torch.ones([latents_pc_.shape[0], text_len]).type_as(encoder_hidden_states_pc[-1])

                text_attention_mask = torch.ones([latents_pc_.shape[0], text_len]).type_as(encoder_hidden_states_pc[-1])
                text_attention_mask = torch.cat([uncond_text_attention_mask, text_attention_mask], dim=0)

                # uncond_text_attention_mask_img = torch.zeros([latents_img_.shape[0], text_len]).type_as(encoder_hidden_states_img[-1])
                # uncond_text_attention_mask_img[:, 0] = True
                uncond_text_attention_mask_img = torch.ones([latents_img_.shape[0], text_len]).type_as(encoder_hidden_states_img[-1])

                text_attention_mask_img = torch.ones([latents_img_.shape[0], text_len]).type_as(encoder_hidden_states_img[-1])
                text_attention_mask_img = torch.cat([uncond_text_attention_mask_img, text_attention_mask_img], dim=0)

                noise_pred_pc, noise_pred_img = self.unet(
                    latent_model_input_pc,
                    latent_model_input_img,
                    t_pc, 
                    t_img, 
                    encoder_hidden_states_pc=encoder_hidden_states_pc,  # b x n, len + 1, 768
                    encoder_hidden_states_img=encoder_hidden_states_img,  # b x n, len + 1, 768
                    attention_mask_pc=text_attention_mask,
                    attention_mask_img=text_attention_mask_img,

                    lidar2imgs=lidar2imgs,
                    # TODO: during training, some camera param are masked.
                )

                noise_pred_pc = noise_pred_pc.sample
                noise_pred_img = noise_pred_img.sample              

                # perform guidance
                if do_classifier_free_guidance:
                    # for each: bxN, 4, 28, 50
                    noise_pred_uncond_pc, noise_pred_text_pc = noise_pred_pc.chunk(2)
                    noise_pred_pc = noise_pred_uncond_pc + guidance_scale * (
                        noise_pred_text_pc - noise_pred_uncond_pc
                    )

                    noise_pred_uncond_img, noise_pred_text_img = noise_pred_img.chunk(2)
                    noise_pred_img = noise_pred_uncond_img + guidance_scale * (
                        noise_pred_text_img - noise_pred_uncond_img
                    )


                # compute the previous noisy sample x_t -> x_t-1
                # NOTE: is the scheduler use randomness, please handle the logic
                # for generator.
                if self.only_point_clouds:
                    latents_pc = self.scheduler_pc.step(
                        noise_pred_pc, t_pc, latents_pc, **extra_step_kwargs
                    ).prev_sample
                elif self.only_images:
                    latents_img = self.scheduler_img.step(
                        noise_pred_img, t_img, latents_img, **extra_step_kwargs
                    ).prev_sample   
                else:
                    latents_pc = self.scheduler_pc.step(
                        noise_pred_pc, t_pc, latents_pc, **extra_step_kwargs
                    ).prev_sample

                    latents_img = self.scheduler_img.step(
                        noise_pred_img, t_img, latents_img, **extra_step_kwargs
                    ).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler_pc.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        ###### BEV: here rebuild the shapes back. post-process still assume
        # latents, no need for b, n, 4, 28, 50
        # prompt_embeds, no need for b, len, 768
        # image, no need for b, c, 200, 200
        ##### BEV end

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            torch.cuda.empty_cache()

        if output_type == "latent":
            image = latents_img
            range_img = latents_img_pc
            has_nsfw_concept = None

        else:
            # 8. Post-processing
            image = self.decode_latents(latents_img, self.vae_img)
            range_img = self.decode_latents(latents_pc, self.vae_pc)
            range_img = range_img.transpose(0, 1, 3, 2)

            image = image / 2 + 0.5

            image = np.clip(image, a_min=0, a_max=1)
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )
            image = image.transpose(0, 2, 3, 1)
            image = self.numpy_to_pil_double([image])

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        image = BEVStableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )
        return range_img, image
