# X-Drive: Cross-modality Consistent Multi-Sensor Data Synthesis for Driving Scenarios

<img width="935" alt="截屏2024-10-31 下午10 58 44" src="https://github.com/user-attachments/assets/40c020a9-58c0-440e-a849-4c950322eade">

## Abstract
Recent advancements have exploited diffusion models for the synthesis of either LiDAR point clouds or camera image data in driving scenarios. Despite their success in modeling single-modality data marginal distribution, there is an under- exploration in the mutual reliance between different modalities to describe com- plex driving scenes. To fill in this gap, we propose a novel framework, X-DRIVE, to model the joint distribution of point clouds and multi-view images via a dual- branch latent diffusion model architecture. Considering the distinct geometrical spaces of the two modalities, X-DRIVE conditions the synthesis of each modality on the corresponding local regions from the other modality, ensuring better alignment and realism. To further handle the spatial ambiguity during denoising, we design the cross-modality condition module based on epipolar lines to adaptively learn the cross-modality local correspondence. Besides, X-DRIVE allows for controllable generation through multi-level input conditions, including text, bounding box, image, and point clouds. Extensive results demonstrate the high-fidelity synthetic results of X-DRIVE for both point clouds and multi-view images, adhering to input conditions while ensuring reliable cross-modality consistency.

[[paper link](http://arxiv.org/abs/2411.01123)]
