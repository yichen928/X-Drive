from typing import Tuple, Union, List
import os
import logging
from hydra.core.hydra_config import HydraConfig
from functools import partial
from omegaconf import OmegaConf
from omegaconf import DictConfig
from PIL import Image
import safetensors

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

from mmdet3d.datasets import build_dataset
from diffusers import UniPCMultistepScheduler
import accelerate
from accelerate.utils import set_seed

from xdrive.dataset import collate_fn, ListSetWrapper, FolderSetWrapper

from xdrive.pipeline.pipeline_bev_controlnet import (
    StableDiffusionBEVControlNetPipeline,
    BEVStableDiffusionPipelineOutput,
)
from xdrive.runner.utils import (
    visualize_map, img_m11_to_01, show_box_on_views
)
from xdrive.misc.common import load_module
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
    UNet2DConditionModel
)
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from xdrive.dataset.utils import multi_bbox_collate_fn
from xdrive.networks.unet_pc_condition_RangeLDM import RangeLDMPCUNet2DModel

from third_party.RangeLDM.ldm.utils import replace_attn, replace_conv, replace_down


def insert_pipeline_item(cfg: DictConfig, search_type, item=None) -> None:
    if item is None:
        return
    assert OmegaConf.is_list(cfg)
    ori_cfg: List = OmegaConf.to_container(cfg)
    for index, _it in enumerate(cfg):
        if _it['type'] == search_type:
            break
    else:
        raise RuntimeError(f"cannot find type: {search_type}")
    ori_cfg.insert(index + 1, item)
    cfg.clear()
    cfg.merge_with(ori_cfg)


def draw_box_on_imgs(cfg, idx, val_input, ori_imgs, transparent_bg=False) -> Tuple[Image.Image, ...]:
    if transparent_bg:
        in_imgs = [Image.new('RGB', img.size) for img in ori_imgs]
    else:
        in_imgs = ori_imgs
    out_imgs = show_box_on_views(
        OmegaConf.to_container(cfg.dataset.object_classes, resolve=True),
        in_imgs,
        val_input['meta_data']['gt_bboxes_3d'][idx].data,
        val_input['meta_data']['gt_labels_3d'][idx].data.numpy(),
        val_input['meta_data']['lidar2image'][idx].data.numpy(),
        val_input['meta_data']['img_aug_matrix'][idx].data.numpy(),
    )
    if transparent_bg:
        for i in range(len(out_imgs)):
            out_imgs[i].putalpha(Image.fromarray(
                (np.any(np.asarray(out_imgs[i]) > 0, axis=2) * 255).astype(np.uint8)))
    return out_imgs


def update_progress_bar_config(pipe, **kwargs):
    if hasattr(pipe, "_progress_bar_config"):
        config = pipe._progress_bar_config
        config.update(kwargs)
    else:
        config = kwargs
    pipe.set_progress_bar_config(**config)


def setup_logger_seed(cfg):
    #### setup logger ####
    # only log debug info to log file
    logging.getLogger().setLevel(logging.DEBUG)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.DEBUG)
        else:
            handler.setLevel(logging.INFO)
    # handle log from some packages
    logging.getLogger("shapely.geos").setLevel(logging.WARN)
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.getLogger("accelerate.tracking").setLevel(logging.INFO)
    logging.getLogger("numba").setLevel(logging.WARN)
    logging.getLogger("PIL").setLevel(logging.WARN)
    logging.getLogger("matplotlib").setLevel(logging.WARN)
    setattr(cfg, "log_root", HydraConfig.get().runtime.output_dir)
    set_seed(cfg.seed)


def build_multi_pipe(cfg, device):
    weight_dtype = torch.float32
    if cfg.resume_from_checkpoint.endswith("/"):
        cfg.resume_from_checkpoint = cfg.resume_from_checkpoint[:-1]

    ##### Load modules
    img_vae = AutoencoderKL.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="vae")
    img_vae = img_vae.to(device)
    img_vae.eval()

    pc_vae_config_path = os.path.join(cfg.model.pretrained_pc_model_path, "vae", "config.json")
    pc_vae_config = AutoencoderKL.load_config(pc_vae_config_path)
    pc_vae = AutoencoderKL.from_config(pc_vae_config)

    pc_vae_checkpoint = safetensors.torch.load_file(os.path.join(cfg.model.pretrained_pc_model_path, "vae", "diffusion_pytorch_model.safetensors"))
    if 'quant_conv.weight' not in pc_vae_checkpoint:
        pc_vae.quant_conv = torch.nn.Identity()
        pc_vae.post_quant_conv = torch.nn.Identity()
    replace_down(pc_vae)
    replace_conv(pc_vae)
    if 'encoder.mid_block.attentions.0.to_q.weight' not in pc_vae_checkpoint:
        replace_attn(pc_vae)
    pc_vae.load_state_dict(pc_vae_checkpoint)

    pc_vae = pc_vae.cuda()
    pc_vae.eval()

    pc_unet_config_path = os.path.join(cfg.model.pretrained_pc_model_path, "unet", "config.json")
    pc_unet_config = RangeLDMPCUNet2DModel.load_config(pc_unet_config_path)
    pc_unet_config['down_block_types'] = ['DownBlock2D', 'CondAttnDownBlock2D', 'CondAttnDownBlock2D', 'CondAttnDownBlock2D']
    pc_unet_config['up_block_types'] = ['CondAttnUpBlock2D', 'CondAttnUpBlock2D', 'CondAttnUpBlock2D', 'UpBlock2D']
    pc_unet_config['cross_attention_dim'] = cfg.model.pc_unet.cross_attention_dim
    pc_unet_config['use_gsa'] = cfg.model.pc_unet.use_gsa

    pc_unet = RangeLDMPCUNet2DModel.from_config(pc_unet_config)

    replace_down(pc_unet)
    replace_conv(pc_unet)

    pc_unet.eval()

    img_unet_class = load_module(cfg.model.img_unet_class)
    pretrain_unet = UNet2DConditionModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="unet")
    img_unet_param = OmegaConf.to_container(cfg.model.img_unet, resolve=True)
    img_unet = img_unet_class.from_unet_2d_condition(pretrain_unet, **img_unet_param)

    unet_class = load_module(cfg.model.unet_class)
    unet = unet_class(img_unet=img_unet, pc_unet=pc_unet, **cfg.model.unet)

    unet_ckpt_dict = torch.load(os.path.join(cfg.resume_from_checkpoint, "pytorch_model.bin"))
    unet.load_state_dict(unet_ckpt_dict)
    unet.eval()

    model_cls = load_module(cfg.model.pc_bbox_embedder_cls)
    pc_bbox_embedder = model_cls(**cfg.model.pc_bbox_embedder_param)
    pc_bbox_embedder_ckpt_dict = torch.load(os.path.join(cfg.resume_from_checkpoint, "pytorch_model_1.bin"))

    pc_bbox_embedder.load_state_dict(pc_bbox_embedder_ckpt_dict)
    pc_bbox_embedder.eval()

    model_cls = load_module(cfg.model.img_bbox_embedder_cls)
    img_bbox_embedder = model_cls(**cfg.model.img_bbox_embedder_param)
    img_bbox_embedder_ckpt_dict = torch.load(os.path.join(cfg.resume_from_checkpoint, "pytorch_model_2.bin"))

    img_bbox_embedder.load_state_dict(img_bbox_embedder_ckpt_dict)
    img_bbox_embedder.eval()

    unet_ckpt_dict = torch.load(os.path.join(cfg.resume_from_checkpoint, "ema_model_0.pth"))
    for s_param, param in zip(unet_ckpt_dict['shadow_params'], unet.parameters()):
        param.data.copy_(s_param.to(param.device).data)

    pc_box_embedder_ckpt_dict = torch.load(os.path.join(cfg.resume_from_checkpoint, "ema_model_1.pth"))
    for s_param, param in zip(pc_box_embedder_ckpt_dict['shadow_params'], pc_bbox_embedder.parameters()):
        param.data.copy_(s_param.to(param.device).data)

    img_box_embedder_ckpt_dict = torch.load(os.path.join(cfg.resume_from_checkpoint, "ema_model_2.pth"))
    for s_param, param in zip(img_box_embedder_ckpt_dict['shadow_params'], img_bbox_embedder.parameters()):
        param.data.copy_(s_param.to(param.device).data)

    tokenizer = CLIPTokenizer.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="tokenizer") 
    text_encoder = CLIPTextModel.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="text_encoder")
    #####

    pipe_param={
        "vae_pc": pc_vae,
        "vae_img": img_vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler_pc": DDPMScheduler.from_pretrained(cfg.model.pretrained_pc_model_path, subfolder="scheduler"),
        "scheduler_img": DDPMScheduler.from_pretrained(cfg.model.pretrained_model_name_or_path, subfolder="scheduler"),
        "unet": unet,
        "bbox_embedder_pc": pc_bbox_embedder,
        "bbox_embedder_img": img_bbox_embedder,
    }

    pipe_cls = load_module(cfg.model.pipe_module)
    logging.info(f"Build pipeline with {pipe_cls}")
    pipe = pipe_cls.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        **pipe_param,
        safety_checker=None,
        feature_extractor=None,  # since v1.5 has default, we need to override
        torch_dtype=weight_dtype
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler_pc = UniPCMultistepScheduler.from_config(
        pipe.scheduler_pc.config
    )
    pipe.scheduler_img = UniPCMultistepScheduler.from_config(
        pipe.scheduler_img.config
    )
    # remove following line if xformers is not installed
    if cfg.runner.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()

    pipe = pipe.to(device)

    # when inference, memory is not the issue. we do not need this.
    # pipe.enable_model_cpu_offload()
    return pipe, weight_dtype

def build_pipe(cfg, device):
    weight_dtype = torch.float16
    if cfg.resume_from_checkpoint.endswith("/"):
        cfg.resume_from_checkpoint = cfg.resume_from_checkpoint[:-1]
    pipe_param = {}

    model_cls = load_module(cfg.model.model_module)
    controlnet_path = os.path.join(
        cfg.resume_from_checkpoint, cfg.model.controlnet_dir)
    logging.info(f"Loading controlnet from {controlnet_path} with {model_cls}")
    controlnet = model_cls.from_pretrained(
        controlnet_path, torch_dtype=weight_dtype)
    controlnet.eval()  # from_pretrained will set to eval mode by default
    pipe_param["controlnet"] = controlnet

    if hasattr(cfg.model, "unet_module"):
        unet_cls = load_module(cfg.model.unet_module)
        unet_path = os.path.join(cfg.resume_from_checkpoint, cfg.model.unet_dir)
        logging.info(f"Loading unet from {unet_path} with {unet_cls}")
        unet = unet_cls.from_pretrained(
            unet_path, torch_dtype=weight_dtype)
        unet.eval()
        pipe_param["unet"] = unet

    pipe_cls = load_module(cfg.model.pipe_module)
    logging.info(f"Build pipeline with {pipe_cls}")
    pipe = pipe_cls.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        **pipe_param,
        safety_checker=None,
        feature_extractor=None,  # since v1.5 has default, we need to override
        torch_dtype=weight_dtype
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed
    if cfg.runner.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()

    pipe = pipe.to(device)

    # when inference, memory is not the issue. we do not need this.
    # pipe.enable_model_cpu_offload()
    return pipe, weight_dtype


def prepare_all(cfg, device='cuda', need_loader=True):
    assert cfg.resume_from_checkpoint is not None, "Please set model to load"
    setup_logger_seed(cfg)

    #### model ####
    pipe, weight_dtype = build_pipe(cfg, device)
    update_progress_bar_config(pipe, leave=False)

    if not need_loader:
        return pipe, weight_dtype

    #### datasets ####

    if cfg.runner.validation_index == "demo":
        val_dataset = FolderSetWrapper("demo/data")
    else:
        val_dataset = build_dataset(
            OmegaConf.to_container(cfg.dataset.data.val, resolve=True)
        )
        if cfg.runner.validation_index != "all":
            val_dataset = ListSetWrapper(
                val_dataset, cfg.runner.validation_index)

    #### dataloader ####
    collate_fn_param = {
        "tokenizer": pipe.tokenizer,
        "template": cfg.dataset.template,
        "bbox_mode": cfg.model.bbox_mode,
        "bbox_view_shared": cfg.model.bbox_view_shared,
        "bbox_drop_ratio": cfg.runner.bbox_drop_ratio,
        "bbox_add_ratio": cfg.runner.bbox_add_ratio,
        "bbox_add_num": cfg.runner.bbox_add_num,
    }
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=partial(collate_fn, is_train=False, **collate_fn_param),
        batch_size=cfg.runner.validation_batch_size,
        num_workers=cfg.runner.num_workers,
    )
    return pipe, val_dataloader, weight_dtype


def prepare_multi_all(cfg, device='cuda', need_loader=True, shuffle_dataloader=False):
    assert cfg.resume_from_checkpoint is not None, "Please set model to load"
    setup_logger_seed(cfg)

    #### model ####
    pipe, weight_dtype = build_multi_pipe(cfg, device)
    update_progress_bar_config(pipe, leave=False)

    if not need_loader:
        return pipe, weight_dtype

    #### datasets ####

    if cfg.runner.validation_index == "demo":
        val_dataset = FolderSetWrapper("demo/data")
    else:
        val_dataset = build_dataset(
            OmegaConf.to_container(cfg.dataset.data.val, resolve=True)
        )
        if cfg.runner.validation_index != "all":
            val_dataset = ListSetWrapper(
                val_dataset, cfg.runner.validation_index)

    #### dataloader ####
    collate_fn_param = {
        # "tokenizer": pipe.tokenizer,
        "template": cfg.dataset.template,
        "bbox_mode": cfg.model.bbox_mode,
        "bbox_view_shared": cfg.model.bbox_view_shared,
        # "bbox_drop_ratio": cfg.runner.bbox_drop_ratio,
        # "bbox_add_ratio": cfg.runner.bbox_add_ratio,
        # "bbox_add_num": cfg.runner.bbox_add_num,
    }
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=shuffle_dataloader,
        collate_fn=partial(multi_bbox_collate_fn, is_train=False, **collate_fn_param),
        batch_size=cfg.runner.validation_batch_size,
        num_workers=cfg.runner.num_workers,
    )
    return pipe, val_dataloader, weight_dtype



def new_local_seed(global_generator):
    local_seed = torch.randint(
        0x7ffffffffffffff0, [1], generator=global_generator).item()
    logging.debug(f"Using seed: {local_seed}")
    return local_seed


def run_one_batch_pipe(
    cfg,
    pipe: StableDiffusionBEVControlNetPipeline,
    pixel_values: torch.FloatTensor,
    captions: Union[str, List[str]],
    bev_map_with_aux: torch.FloatTensor,
    camera_param: Union[torch.Tensor, None],
    bev_controlnet_kwargs: dict,
    global_generator=None
):
    """call pipe several times to generate images

    Args:
        cfg (_type_): _description_
        pipe (StableDiffusionBEVControlNetPipeline): _description_
        captions (Union[str, List[str]]): _description_
        bev_map_with_aux (torch.FloatTensor): (B=1, C=26, 200, 200), float32
        camera_param (Union[torch.Tensor, None]): (B=1, N=6, 3, 7), if None, 
            use learned embedding for uncond_cam

    Returns:
        List[List[List[Image.Image]]]: 3-dim list of PIL Image: B, Times, views
    """
    # for each input param, we generate several times to check variance.
    if isinstance(captions, str):
        batch_size = 1
    else:
        batch_size = len(captions)

    # let different prompts have the same random seed
    if cfg.seed is None:
        generator = None
    else:
        if global_generator is not None:
            if cfg.fix_seed_within_batch:
                generator = []
                for _ in range(batch_size):
                    local_seed = new_local_seed(global_generator)
                    generator.append(torch.manual_seed(local_seed))
            else:
                local_seed = new_local_seed(global_generator)
                generator = torch.manual_seed(local_seed)
        else:
            if cfg.fix_seed_within_batch:
                generator = [torch.manual_seed(cfg.seed)
                             for _ in range(batch_size)]
            else:
                generator = torch.manual_seed(cfg.seed)

    gen_imgs_list = [[] for _ in range(batch_size)]
    for ti in range(cfg.runner.validation_times):
        image: BEVStableDiffusionPipelineOutput = pipe(
            prompt=captions,
            image=bev_map_with_aux,
            camera_param=camera_param,
            height=cfg.dataset.image_size[0],
            width=cfg.dataset.image_size[1],
            generator=generator,
            bev_controlnet_kwargs=bev_controlnet_kwargs,
            **cfg.runner.pipeline_param,
        )
        image: List[List[Image.Image]] = image.images
        for bi, imgs in enumerate(image):
            gen_imgs_list[bi].append(imgs)
    return gen_imgs_list


def run_one_batch(cfg, pipe, val_input, weight_dtype, global_generator=None,
                  run_one_batch_pipe_func=run_one_batch_pipe,
                  transparent_bg=False, map_size=400):
    """Run one batch of data according to your configuration

    Returns:
        List[Image.Image]: map image
        List[List[Image.Image]]: ori images
        List[List[Image.Image]]: ori images with bbox, can be []
        List[List[Tuple[Image.Image]]]: generated images list
        List[List[Tuple[Image.Image]]]: generated images list, can be []
        if 2-dim: B, views; if 3-dim: B, Times, views
    """
    bs = len(val_input['meta_data']['metas'])

    # TODO: not sure what to do with filenames
    # image_names = [val_input['meta_data']['metas'][i].data['filename']
    #                for i in range(bs)]
    logging.debug(f"Caption: {val_input['captions']}")

    # map
    map_imgs = []
    for bev_map in val_input["bev_map_with_aux"]:
        map_img_np = visualize_map(cfg, bev_map, target_size=map_size)
        map_imgs.append(Image.fromarray(map_img_np))

    # ori
    ori_imgs = [None for bi in range(bs)]
    ori_imgs_with_box = [None for bi in range(bs)]
    if val_input["pixel_values"] is not None:
        ori_imgs = [
            [to_pil_image(img_m11_to_01(val_input["pixel_values"][bi][i]))
             for i in range(6)] for bi in range(bs)
        ]
        if cfg.show_box:
            ori_imgs_with_box = [
                draw_box_on_imgs(cfg, bi, val_input, ori_imgs[bi],
                                 transparent_bg=transparent_bg)
                for bi in range(bs)
            ]

    # camera_emb = self._embed_camera(val_input["camera_param"])
    camera_param = val_input["camera_param"].to(weight_dtype)

    # 3-dim list: B, Times, views

    gen_imgs_list = run_one_batch_pipe_func(
        cfg, pipe, val_inputs['pixel_values'], val_input['captions'],
        val_input['bev_map_with_aux'], camera_param, val_input['kwargs'],
        global_generator=global_generator)

    # save gen with box
    gen_imgs_wb_list = []
    if cfg.show_box:
        for bi, images in enumerate(gen_imgs_list):
            gen_imgs_wb_list.append([
                draw_box_on_imgs(cfg, bi, val_input, images[ti],
                                 transparent_bg=transparent_bg)
                for ti in range(len(images))
            ])
    else:
        for bi, images in enumerate(gen_imgs_list):
            gen_imgs_wb_list.append(None)

    return map_imgs, ori_imgs, ori_imgs_with_box, gen_imgs_list, gen_imgs_wb_list


def run_multi_one_batch_pipe(
    cfg,
    pipe: StableDiffusionBEVControlNetPipeline,
    pixel_values: torch.FloatTensor, 
    range_image: torch.FloatTensor,
    captions: Union[str, List[str]],
    lidar2imgs: Union[torch.Tensor, None],
    camera_param: Union[torch.Tensor, None],
    val_data_kwargs: dict,
    # image_only=False,
    # pc_only=False,
    wo_boxes=False,
    wo_texts=False,
    global_generator=None
):
    """call pipe several times to generate images

    Args:
        cfg (_type_): _description_
        pipe (StableDiffusionBEVControlNetPipeline): _description_
        captions (Union[str, List[str]]): _description_
        bev_map_with_aux (torch.FloatTensor): (B=1, C=26, 200, 200), float32
        camera_param (Union[torch.Tensor, None]): (B=1, N=6, 3, 7), if None, 
            use learned embedding for uncond_cam

    Returns:
        List[List[List[Image.Image]]]: 3-dim list of PIL Image: B, Times, views
    """
    # for each input param, we generate several times to check variance.
    if isinstance(captions, str):
        batch_size = 1
    else:
        batch_size = len(captions)

    # let different prompts have the same random seed
    if cfg.seed is None:
        generator = None
    else:
        if global_generator is not None:
            if cfg.fix_seed_within_batch:
                generator = []
                for _ in range(batch_size):
                    local_seed = new_local_seed(global_generator)
                    generator.append(torch.manual_seed(local_seed))
            else:
                local_seed = new_local_seed(global_generator)
                generator = torch.manual_seed(local_seed)
        else:
            if cfg.fix_seed_within_batch:
                generator = [torch.manual_seed(cfg.seed)
                             for _ in range(batch_size)]
            else:
                generator = torch.manual_seed(cfg.seed)

    latents_pc = None
    latents_img = None
    if range_image is not None:
        pc_vae = pipe.vae_pc
        latents_pc = pc_vae.encode(range_image).latent_dist.sample()
        latents_pc = latents_pc * pc_vae.config.scaling_factor
        pipe.only_images = True
    if pixel_values is not None:
        img_vae = pipe.vae_img
        latents_img = img_vae.encode(pixel_values.reshape(-1, pixel_values.shape[2], pixel_values.shape[3], pixel_values.shape[4])).latent_dist.sample()
        latents_img = latents_img * img_vae.config.scaling_factor
        pipe.only_point_clouds = True
    
    if wo_boxes:
        pipe.wo_boxes = True
    if wo_texts:
        pipe.wo_texts = True

    gen_imgs_list = [[] for _ in range(batch_size)]
    gen_range_imgs_list = []
    for ti in range(cfg.runner.validation_times):
        range_image, image = pipe(
            prompt=captions,
            latents_pc=latents_pc,
            latents_img=latents_img,
            camera_param=camera_param,
            range_height=cfg.dataset.range_img_size[0],
            range_width=cfg.dataset.range_img_size[1],
            width=cfg.dataset.image_size[1],
            height=cfg.dataset.image_size[0],
            generator=generator,
            lidar2imgs=lidar2imgs,
            val_data_kwargs=val_data_kwargs,
            **cfg.runner.pipeline_param,
        )
        image: List[List[Image.Image]] = image.images
        for bi, imgs in enumerate(image):
            gen_imgs_list[bi].append(imgs)
        
        gen_range_imgs_list.append(range_image)

    return gen_range_imgs_list, gen_imgs_list


def run_multi_one_batch(cfg, pipe, val_input, weight_dtype, global_generator=None,
                  run_multi_one_batch_pipe_func=run_multi_one_batch_pipe,
                  image_only=False, pc_only=False, wo_boxes=False, wo_texts=False,
                  transparent_bg=False, map_size=400):
    """Run one batch of data according to your configuration

    Returns:
        List[Image.Image]: map image
        List[List[Image.Image]]: ori images
        List[List[Image.Image]]: ori images with bbox, can be []
        List[List[Tuple[Image.Image]]]: generated images list
        List[List[Tuple[Image.Image]]]: generated images list, can be []
        if 2-dim: B, views; if 3-dim: B, Times, views
    """
    bs = len(val_input['meta_data']['metas'])

    # TODO: not sure what to do with filenames
    # image_names = [val_input['meta_data']['metas'][i].data['filename']
    #                for i in range(bs)]
    logging.debug(f"Caption: {val_input['captions']}")

    # ori
    ori_imgs = [None for bi in range(bs)]
    ori_imgs_with_box = [None for bi in range(bs)]
    if val_input["pixel_values"] is not None:
        ori_imgs = [
            [to_pil_image(img_m11_to_01(val_input["pixel_values"][bi][i]))
             for i in range(6)] for bi in range(bs)
        ]
        if cfg.show_box:
            ori_imgs_with_box = [
                draw_box_on_imgs(cfg, bi, val_input, ori_imgs[bi],
                                 transparent_bg=transparent_bg)
                for bi in range(bs)
            ]

    # camera_emb = self._embed_camera(val_input["camera_param"])
    camera_param = val_input["camera_param"].to(weight_dtype)
    lidar2imgs = val_input["lidar2imgs"].to(weight_dtype)

    if image_only:
        assert not pc_only
        pixel_values = None
        range_img = val_input['range_img']
    elif pc_only:
        pixel_values = val_input['pixel_values']
        range_img = None
    else:
        pixel_values = None
        range_img = None

    # 3-dim list: B, Times, views
    gen_range_imgs_list, gen_imgs_list = run_multi_one_batch_pipe_func(
        cfg, pipe, pixel_values, range_img, val_input['captions'],
        lidar2imgs, camera_param, val_data_kwargs=val_input['kwargs'],
        wo_boxes=wo_boxes, wo_texts=wo_texts,
        global_generator=global_generator)

    # save gen with box
    gen_imgs_wb_list = []
    if cfg.show_box:
        for bi, images in enumerate(gen_imgs_list):
            gen_imgs_wb_list.append([
                draw_box_on_imgs(cfg, bi, val_input, images[ti],
                                 transparent_bg=transparent_bg)
                for ti in range(len(images))
            ])
    else:
        for bi, images in enumerate(gen_imgs_list):
            gen_imgs_wb_list.append(None)

    return ori_imgs, ori_imgs_with_box, gen_imgs_list, gen_imgs_wb_list, gen_range_imgs_list