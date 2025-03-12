# [ICLR 2025] X-Drive: Cross-modality Consistent Multi-Sensor Data Synthesis for Driving Scenarios

## Abstract
Recent advancements have exploited diffusion models for the synthesis of either LiDAR point clouds or camera image data in driving scenarios. Despite their success in modeling single-modality data marginal distribution, there is an under- exploration in the mutual reliance between different modalities to describe complex driving scenes. To fill in this gap, we propose a novel framework, X-DRIVE, to model the joint distribution of point clouds and multi-view images via a dual- branch latent diffusion model architecture. Considering the distinct geometrical spaces of the two modalities, X-DRIVE conditions the synthesis of each modality on the corresponding local regions from the other modality, ensuring better alignment and realism. To further handle the spatial ambiguity during denoising, we design the cross-modality condition module based on epipolar lines to adaptively learn the cross-modality local correspondence. Besides, X-DRIVE allows for controllable generation through multi-level input conditions, including text, bounding box, image, and point clouds. Extensive results demonstrate the high-fidelity synthetic results of X-DRIVE for both point clouds and multi-view images, adhering to input conditions while ensuring reliable cross-modality consistency.

[[paper link](http://arxiv.org/abs/2411.01123)]

## Framework
We jointly generate pairwise LiDAR-camera data with cross-modality consistency.
<img width="935" alt="截屏2024-10-31 下午10 58 44" src="https://github.com/user-attachments/assets/40c020a9-58c0-440e-a849-4c950322eade">

## Qualitative Results
<img width="1133" alt="results" src="https://github.com/user-attachments/assets/d5b030d7-1150-4d73-8472-fb378eeb4fd0" />

## Improvement compared to the first ArXiv version
+ We incorporate [RangeLDM](https://github.com/WoodwindHu/RangeLDM) model architecture and pretrained weights into our LiDAR branch to simplify our training pipleine. We thank the authors for releasing their excellent work.
+ We include the EMA model in our training pipeline.

## Updates

- [x] Training code
- [ ] DAS metric for multimodal alignment
- [ ] Visualization code
- [ ] Pretrained checkpoints & Generation of synthetic dataset

## Getting start
Our code base is developed on the basis of [MagicDrive](https://github.com/cure-lab/MagicDrive), so our enviornment setup is almost same as them. We appreciate their efforts in making the code open-source!

### Environment Setup
The code is tested with Pytorch==1.10.2 on A6000 GPUs. To set up the python environment, follow:

`pip install -r requirements/dev.txt`

We opt to install the source code for the following packages, with `cd ${FOLDER}; pip -vvv install .`
```bash
# install third-party
third_party/
├── bevfusion -> based on db75150
├── diffusers -> based on v0.17.1 (afcca39)
└── xformers  -> based on v0.0.19 (8bf59c9), optional
```

see [note about our xformers](doc/xformers.md). If you have issues with the environment setup, please check [FAQ](doc/FAQ.md) first.


### Data Preparation
We prepare the nuScenes dataset similar to [MagicDrive]([https://github.com/mit-han-lab/bevfusion#data-preparation](https://github.com/cure-lab/MagicDrive)). Specifically,

1. Download the nuScenes dataset from the [website](https://www.nuscenes.org/nuscenes) and put them in `./data/`. You should have these files:
    ```bash
    data/nuscenes
    ├── maps
    ├── mini
    ├── samples
    ├── sweeps
    ├── v1.0-mini
    └── v1.0-trainval
    ```
    
2. Prepare the mmdet3d annotation file
> [!TIP]
> You can download the `.pkl` files from [Google Drive](https://drive.google.com/drive/folders/1C0TonFSvJM0Lf39Rj2KykPaCz4qyuSRF?usp=sharing).

Or alternatively, you can generate mmdet3d annotation files by:

    ```bash
    python tools/create_data.py nuscenes --root-path ./data/nuscenes \
      --out-dir ./data/nuscenes_mmdet3d_2 --extra-tag nuscenes
    ```
    You should have these files:
    ```bash
    data/nuscenes_mmdet3d_2
    ├── nuscenes_dbinfos_train.pkl (-> ${bevfusion-version}/nuscenes_dbinfos_train.pkl)
    ├── nuscenes_gt_database (-> ${bevfusion-version}/nuscenes_gt_database)
    ├── nuscenes_infos_train.pkl
    └── nuscenes_infos_val.pkl
    ```

## Training Pipeline
### Initial Weights
We initialize our multi-modal generation model with [stable-diffusion-2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) (for camera branch) and [RangeLDM](https://drive.google.com/drive/folders/1rP0_YNgBn-BgqreJ8p--nyaEEGS3eofs?usp=sharing) (for LiDAR branch). Please download them and make the directory as follow:

```bash
X-Drive
├── pretrained
    ├── stable-diffusion-2-1-base
    └── RangeLDM-nuScenes
```



### Model Training
Our method requires two stages for training:

1. In the first stage, we train the diffusion model for single-modality point clouds data conditioned on text description and 3D box

Launch training (with 4xA6000 GPUs):
```bash
accelerate launch --mixed_precision fp16 --gpu_ids 0,1,2,3 --num_processes 4 tools/train.py --config-name=config_pc_RangeLDM_box  +exp=pc_ldm_box runner=4gpus_pc
```

2. In the second stage, we train the multi-modal diffusion model from the pretrained LiDAR branch and camera branch

Launch training (with 4xA6000 GPUs):
```bash
accelerate launch --mixed_precision fp16 --gpu_ids 0,1,2,3 --num_processes 4 tools/train.py --config-name=config_multi_box  +exp=multi_ldm_box runner=4gpus_multi
```
During training, you can check tensorboard for the log and intermediate results.

## Model Inference and Visualization
Coming soon...

## Reference
If our paper or code are helpful, please consider citing it as
```bash
@article{xie2024x,
  title={X-Drive: Cross-modality consistent multi-sensor data synthesis for driving scenarios},
  author={Xie, Yichen and Xu, Chenfeng and Peng, Chensheng and Zhao, Shuqi and Ho, Nhat and Pham, Alexander T and Ding, Mingyu and Tomizuka, Masayoshi and Zhan, Wei},
  journal={arXiv preprint arXiv:2411.01123},
  year={2024}
}
```


