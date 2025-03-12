from typing import Tuple, List
import logging
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
import torchvision
from torchvision.transforms.functional import to_pil_image, to_tensor

from diffusers import UniPCMultistepScheduler
from accelerate.tracking import GeneralTracker

from xdrive.runner.utils import (
    visualize_map,
    img_m11_to_01,
    concat_6_views,
)
from xdrive.misc.common import move_to
from xdrive.misc.test_utils import draw_box_on_imgs
from xdrive.dataset.utils import multi_bbox_collate_fn
from xdrive.runner.pc_utils import unnormalize_range_img

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def format_ori_with_gen(ori_img, gen_img_list):
    formatted_images = []

    # first image is input, followed by generations.
    formatted_images.append(np.asarray(ori_img))

    for image in gen_img_list:
        formatted_images.append(np.asarray(image))

    # formatted_images = np.stack(formatted_images)
    # 0-255 np -> 0-1 tensor -> grid -> 0-255 pil -> np
    formatted_images = torchvision.utils.make_grid(
        [to_tensor(im) for im in formatted_images], nrow=1)
    formatted_images = np.asarray(
        to_pil_image(formatted_images))
    return formatted_images


class MultiValidatorBox:
    def __init__(self, cfg, val_dataset, pipe_cls, pipe_param) -> None:
        self.cfg = cfg
        self.val_dataset = val_dataset
        self.pipe_cls = pipe_cls
        self.pipe_param = pipe_param
        logging.info(
            f"[MultiValidatorBox] Validator use model_param: {pipe_param.keys()}")

    def validate(
        self,
        bbox_embedder_pc,
        bbox_embedder_img,
        unet,
        trackers: Tuple[GeneralTracker, ...],
        step, weight_dtype, device
    ):
        logging.info("[MultiValidatorBox] Running validation... ")
        bbox_embedder_pc.eval()  # important !!!
        bbox_embedder_img.eval()  # important !!!
        unet.eval()

        pipeline = self.pipe_cls.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path,
            **self.pipe_param,
            unet=unet,
            bbox_embedder_pc=bbox_embedder_pc,
            bbox_embedder_img=bbox_embedder_img,
            safety_checker=None,
            feature_extractor=None,  # since v1.5 has default, we need to override
            torch_dtype=weight_dtype,
        )
        # NOTE: this scheduler does not take generator as kwargs.
        pipeline.scheduler_pc = UniPCMultistepScheduler.from_config(
            pipeline.scheduler_pc.config
        )
        pipeline.scheduler_img = UniPCMultistepScheduler.from_config(
            pipeline.scheduler_img.config
        )
        pipeline = pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)

        if self.cfg.runner.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        image_logs = []
        progress_bar = tqdm(
            range(
                0,
                len(self.cfg.runner.validation_index)
                * self.cfg.runner.validation_times,
            ),
            desc="Val Steps",
        )

        for validation_i in self.cfg.runner.validation_index:
            raw_data = self.val_dataset[validation_i]  # cannot index loader
            val_input = multi_bbox_collate_fn(
                [raw_data], self.cfg.dataset.template, is_train=False,
                bbox_mode=self.cfg.model.bbox_mode,
                bbox_view_shared=self.cfg.model.bbox_view_shared,
            )
            val_input['kwargs']['bboxes_3d_data'] = val_input['kwargs']['bboxes_3d_data_img']
            # camera_emb = self._embed_camera(val_input["camera_param"])
            camera_param = val_input["camera_param"].to(weight_dtype)
            lidar2imgs = val_input["lidar2imgs"].to(weight_dtype)

            # let different prompts have the same random seed
            if self.cfg.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device).manual_seed(
                    self.cfg.seed
                )

            # for each input param, we generate several times to check variance.
            gen_pc_list, gen_img_list, gen_img_wb_list = [], [], []
            for _ in range(self.cfg.runner.validation_times):
                with torch.autocast("cuda"):
                    gen_pc, gen_img = pipeline(
                        prompt=val_input["captions"],
                        camera_param=camera_param,
                        range_height=self.cfg.dataset.range_img_size[0],
                        range_width=self.cfg.dataset.range_img_size[1],
                        width=self.cfg.dataset.image_size[1],
                        height=self.cfg.dataset.image_size[0],
                        generator=generator,
                        val_data_kwargs=val_input["kwargs"],
                        lidar2imgs=lidar2imgs,
                        **self.cfg.runner.pipeline_param,
                    )
                
                gen_pc = gen_pc[0].transpose(1, 2, 0)

                range_img = val_input['range_img']
                range_img = range_img[0].permute(1, 2, 0)
                range_img = range_img.numpy()

                mean = self.cfg.dataset.range_img.mean
                std = self.cfg.dataset.range_img.std
                log_scale = self.cfg.dataset.range_img.log_scale
                max_depth = self.cfg.dataset.range_img.max_depth
                gen_pc = unnormalize_range_img(gen_pc, mean, std, to_01=True, max_depth=max_depth, log_scale=log_scale)
                range_img = unnormalize_range_img(range_img, mean, std, to_01=True, max_depth=max_depth, log_scale=log_scale)

                gen_pc_r = gen_pc[..., 0]
                gen_pc_i = gen_pc[..., 1]

                range_img_r = range_img[..., 0]
                range_img_i = range_img[..., 1]

                norm = plt.Normalize()
                range_img_r = plt.cm.viridis(norm(range_img_r))
                range_img_r = range_img_r[..., :3]

                norm = plt.Normalize()
                range_img_i = plt.cm.viridis(norm(range_img_i))
                range_img_i = range_img_i[..., :3]


                norm = plt.Normalize()
                gen_pc_r = plt.cm.viridis(norm(gen_pc_r))
                gen_pc_r = gen_pc_r[..., :3]

                norm = plt.Normalize()
                gen_pc_i = plt.cm.viridis(norm(gen_pc_i))
                gen_pc_i = gen_pc_i[..., :3]

                ori_pc = np.concatenate([range_img_r, np.ones_like(range_img_r), range_img_i, np.ones_like(range_img_i)], axis=0)
                output_img = np.concatenate([gen_pc_r, np.ones_like(gen_pc_r), gen_pc_i, np.ones_like(gen_pc_i)], axis=0)

                gen_pc_list.append(output_img)

                image = gen_img.images[0]
                gen_img = concat_6_views(image)
                gen_img_list.append(gen_img)

                if self.cfg.runner.validation_show_box:
                    image_with_box = draw_box_on_imgs(
                        self.cfg, 0, val_input, image)
                    gen_img_wb_list.append(concat_6_views(image_with_box))

                progress_bar.update(1)

            ori_imgs = [
                to_pil_image(img_m11_to_01(val_input["pixel_values"][0][i]))
                for i in range(6)
            ]
            ori_img = concat_6_views(ori_imgs)
            ori_img_wb = concat_6_views(draw_box_on_imgs(self.cfg, 0, val_input, ori_imgs))

            image_logs.append(
                {
                    "ori_pc": ori_pc,
                    "ori_img": ori_img,  # input
                    "ori_img_wb": ori_img_wb,  # input
                    "gen_pc_list": gen_pc_list,  # output
                    "gen_img_list": gen_img_list,  # output
                    "gen_img_wb_list": gen_img_wb_list,  # output

                    "validation_prompt": val_input["captions"][0],
                }
            )

        for tracker in trackers:
            if tracker.name == "tensorboard":
                for log in image_logs:
                    validation_prompt = log["validation_prompt"]

                    formatted_pcs = format_ori_with_gen(
                        log["ori_pc"], log["gen_pc_list"])
                    tracker.writer.add_image(
                        validation_prompt + "(pc)", formatted_pcs, step,
                        dataformats="HWC")

                    formatted_images = format_ori_with_gen(log["ori_img"], log["gen_img_list"])
                    tracker.writer.add_image(
                        validation_prompt + "(image)", formatted_images, step,
                        dataformats="HWC")

                    formatted_images = format_ori_with_gen(log["ori_img_wb"], log["gen_img_wb_list"])
                    tracker.writer.add_image(
                        validation_prompt + "(image with box)", formatted_images,
                        step, dataformats="HWC")


            else:
                raise NotImplementedError("Do not use wandb.")

        return image_logs