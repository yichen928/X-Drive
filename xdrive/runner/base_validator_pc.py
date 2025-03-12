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

from xdrive.dataset.utils import lidar_collate_fn
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


class BaseValidatorPC:
    def __init__(self, cfg, val_dataset, pipe_cls, pipe_param) -> None:
        self.cfg = cfg
        self.val_dataset = val_dataset
        self.pipe_cls = pipe_cls
        self.pipe_param = pipe_param
        logging.info(
            f"[PCValidator] Validator use model_param: {pipe_param.keys()}")

    def validate(
        self,
        controlnet,
        unet,
        trackers: Tuple[GeneralTracker, ...],
        step, weight_dtype, device
    ):
        logging.info("[PCValidator] Running validation... ")
        controlnet.eval()  # important !!!
        unet.eval()

        pipeline = self.pipe_cls.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path,
            **self.pipe_param,
            unet=unet,
            controlnet=controlnet,
            safety_checker=None,
            feature_extractor=None,  # since v1.5 has default, we need to override
            torch_dtype=weight_dtype,
        )
        # NOTE: this scheduler does not take generator as kwargs.
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config
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
            val_input = lidar_collate_fn(
                [raw_data], self.cfg.dataset.template, is_train=False,
            )
            # camera_emb = self._embed_camera(val_input["camera_param"])

            # let different prompts have the same random seed
            if self.cfg.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device).manual_seed(
                    self.cfg.seed
                )

            # for each input param, we generate several times to check variance.
            gen_list, gen_wb_list = [], []
            for _ in range(self.cfg.runner.validation_times):
                with torch.autocast("cuda"):
                    gen_img = pipeline(
                        prompt=val_input["captions"],
                        range_height=32,
                        range_width=1024,
                        generator=generator,
                        bev_controlnet_kwargs=val_input["kwargs"],
                        **self.cfg.runner.pipeline_param,
                    )
                
                gen_img = gen_img[0].transpose(1, 2, 0)

                range_img = val_input['range_img']
                range_img = range_img[0].permute(1, 2, 0)
                range_img = range_img.numpy()

                mean = self.cfg.dataset.range_img.mean
                std = self.cfg.dataset.range_img.std

                gen_img = unnormalize_range_img(gen_img, mean, std, to_01=True)
                range_img = unnormalize_range_img(range_img, mean, std, to_01=True)

                gen_img_r = gen_img[..., 0]
                gen_img_i = gen_img[..., 1]

                range_img_r = range_img[..., 0]
                range_img_i = range_img[..., 1]

                norm = plt.Normalize()
                range_img_r = plt.cm.viridis(norm(range_img_r))
                range_img_r = range_img_r[..., :3]

                norm = plt.Normalize()
                range_img_i = plt.cm.viridis(norm(range_img_i))
                range_img_i = range_img_i[..., :3]


                norm = plt.Normalize()
                gen_img_r = plt.cm.viridis(norm(gen_img_r))
                gen_img_r = gen_img_r[..., :3]

                norm = plt.Normalize()
                gen_img_i = plt.cm.viridis(norm(gen_img_i))
                gen_img_i = gen_img_i[..., :3]

                ori_img = np.concatenate([range_img_r, np.ones_like(range_img_r), range_img_i, np.ones_like(range_img_i)], axis=0)
                output_img = np.concatenate([gen_img_r, np.ones_like(gen_img_r), gen_img_i, np.ones_like(gen_img_i)], axis=0)

                gen_list.append(output_img)

                progress_bar.update(1)

            image_logs.append(
                {
                    "ori_img": ori_img,
                    "gen_img_list": gen_list,  # output
                    "validation_prompt": val_input["captions"][0],
                }
            )

        for tracker in trackers:
            if tracker.name == "tensorboard":
                for log in image_logs:
                    validation_prompt = log["validation_prompt"]

                    formatted_images = format_ori_with_gen(
                        log["ori_img"], log["gen_img_list"])
                    tracker.writer.add_image(
                        validation_prompt, formatted_images, step,
                        dataformats="HWC")

            else:
                raise NotImplementedError("Do not use wandb.")

        return image_logs