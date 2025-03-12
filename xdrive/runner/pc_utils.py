import numpy as np
import cv2
import copy
from PIL import Image


def project_points_to_images(images, points, lidar2imgs, max_depth=80):
    images = copy.deepcopy(images)
    points = points[:, :3]
    points_pad = np.ones_like(points[:, :1])
    points_pad = np.concatenate([points, points_pad], axis=-1)
    num_views = lidar2imgs.shape[0]

    new_images = []
    for vid in range(num_views):
        img = np.array(images[vid])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        points_view = np.matmul(lidar2imgs[vid][None], points_pad[..., None]).squeeze(-1)
        points_view[:, :2] = points_view[:, :2] / (points_view[:, 2:3] + 1e-4)

        height, width = img.shape[:2]
        range_filter = (points_view[:, 0] > 0) & (points_view[:, 0] < width) & (points_view[:, 1] > 0) & (points_view[:, 1] < height) & (points_view[:, 2] > 0)
        points_view = points_view[range_filter]
        points_view = points_view[:, :3]

        depth_filter = (points_view[:, 2] < max_depth) & (points_view[:, 2] > 0)
        points_view = points_view[depth_filter]

        depth = (points_view[:, 2]/max_depth*255).astype(np.uint8)
        points_view = points_view.astype(np.int32)

        points_color = cv2.applyColorMap(depth[:, None], cv2.COLORMAP_JET).squeeze(axis=1)

        for pid in range(points_view.shape[0]):
            img = cv2.circle(img, (points_view[pid, 0], points_view[pid, 1]), 1, (int(points_color[pid,0]), int(points_color[pid,1]), int(points_color[pid,2])), -1)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        new_images.append(Image.fromarray(img))

    return new_images




def color_point_clouds(points, BGR_color=False):
    max_int = np.percentile(points[:, 3], 95)
    if max_int > 2:
        points[..., 3] = points[..., 3] / 255
    points[..., 3] = np.clip(points[..., 3], a_min=0, a_max=1)
    max_int = np.percentile(points[:, 3], 95)
    color_map = points[..., 3]
    color_map = color_map / max_int * 255
    color_map = color_map.astype(np.uint8)

    color_map = cv2.applyColorMap(color_map[:, None], cv2.COLORMAP_VIRIDIS).squeeze(axis=1)
    color_map = color_map.astype(np.float32) / 256

    if not BGR_color:
        color_map = color_map[..., ::-1]  # BGR to RGB
    
    return color_map



def unnormalize_range_img(range_img, mean, std, to_01=True, log_scale=False, max_depth=70):
    mean = np.array(mean)
    std = np.array(std)
    range_img = range_img * std[None, None] + mean[None, None]
    if log_scale:
        range_img[..., 0] = np.exp(range_img[..., 0]) - 1
    
    if to_01:
        maximum = np.max(range_img, axis=(0, 1), keepdims=True)
        # if maximum[0] > max_depth:
        #     maximum[0] = max_depth
        #     range_img[..., 1] = np.where(range_img[..., 0]>max_depth, np.zeros_like(range_img[..., 1]), range_img[..., 1])
        range_img = range_img / (maximum + 1e-4)
        range_img = np.clip(range_img, a_min=0, a_max=1)

    return range_img


def unproject_range_img(range_img, mean, std, fov_min=-30, fov_max=10, log_scale=False, filter_points=True):

    mean = np.array(mean)
    std = np.array(std)
    range_img = range_img * std[None, None] + mean[None, None]
    if log_scale:
        range_img[..., 0] = np.exp(range_img[..., 0])

    height = range_img.shape[0]
    width = range_img.shape[1]

    proj_y = np.arange(height) + 0.5
    proj_y = fov_max - (fov_max - fov_min) / height  * proj_y 
    proj_y = proj_y / 180 * np.pi

    proj_x = np.arange(width) + 0.5
    proj_x = 360 / width * proj_x - 180
    proj_x = proj_x / 180 * np.pi

    yaw, pitch = np.meshgrid(proj_x, proj_y, indexing='xy')

    points_range = range_img[..., 0]
    points_intensity = range_img[..., 1]

    points_beam = height - 1 - np.arange(height)
    points_beam = points_beam[:, None].repeat(width, axis=1)

    points_range = np.clip(points_range, a_min=0, a_max=None)
    points_intensity = np.clip(points_intensity, a_min=0, a_max=255)

    points_z = points_range * np.sin(pitch)
    points_y = points_range * np.cos(pitch) * np.sin(yaw)
    points_x = points_range * np.cos(pitch) * np.cos(yaw)

    points = np.stack([points_x, points_y, points_z, points_intensity, points_beam], axis=-1)
    points = points.reshape(-1, 5)

    points_range = points_range.reshape(-1)
    intensity_mask = points[..., 3] >= 1
    points_filter = np.logical_or(np.abs(points[:, 0]) > 2, np.abs(points[:, 1]) > 3)

    if filter_points:
        points_filter = np.logical_and(intensity_mask, points_filter)
        points = points[points_filter]

    return points



def unproject_range_img_RangeLDM(range_img, mean, std, fov_min=-30, fov_max=10, log_scale=False, filter_points=True):

    height_t = np.array([-0.00216031, -0.00098729, -0.00020528,  0.00174976,  0.0044868 , -0.00294233,
            -0.00059629, -0.00020528,  0.00174976, -0.00294233, -0.0013783 ,  0.00018573,
            0.00253177, -0.00098729,  0.00018573,  0.00096774, -0.00411535, -0.0013783,
            0.00018573,  0.00018573, -0.00294233, -0.0013783 , -0.00098729, -0.00020528,
            0.00018573,  0.00018573,  0.00018573, -0.00020528,  0.00018573,  0.00018573,
            0.00018573,  0.00018573,], dtype=np.float32)
    zenith_t = np.array([ 1.86705767e-01,  1.63245357e-01,  1.39784946e-01,  1.16324536e-01,
            9.28641251e-02,  7.01857283e-02,  4.67253177e-02,  2.32649071e-02,
            -1.95503421e-04, -2.28739003e-02, -4.63343109e-02, -6.97947214e-02,
            -9.32551320e-02, -1.15933529e-01, -1.39393939e-01, -1.62854350e-01,
            -1.85532747e-01, -2.08993157e-01, -2.32453568e-01, -2.55913978e-01,
            -2.78592375e-01, -3.02052786e-01, -3.25513196e-01, -3.48973607e-01,
            -3.72434018e-01, -3.95894428e-01, -4.19354839e-01, -4.42033236e-01,
            -4.65493646e-01, -4.88954057e-01, -5.12414467e-01, -5.35874878e-01,], dtype=np.float32)


    mean = np.array(mean)
    std = np.array(std)
    range_img = range_img * std[None, None] + mean[None, None]
    if log_scale:
        range_img[..., 0] = np.exp(range_img[..., 0])


    points_range = range_img[..., 0]
    points_range = np.clip(points_range, a_min=0, a_max=None)
    points_intensity = range_img[..., 1]
    points_intensity = np.clip(points_intensity, a_min=0, a_max=255)
    points_intensity = points_intensity.reshape(-1)

    incl_t = -zenith_t
    theta = np.pi / 2 - incl_t

    height = range_img.shape[0]
    width = range_img.shape[1]

    z = (height_t[:, None] - points_range * np.sin(incl_t[:, None])).reshape(-1)

    xy_norm = points_range * np.cos(incl_t[:, None])
    azi = (width - 0.5 - np.arange(0, width)) / width * 2. * np.pi - np.pi

    x = (xy_norm * np.cos(azi[None, :])).reshape(-1)
    y = (xy_norm * np.sin(azi[None, :])).reshape(-1)

    points = np.stack([x,y,z, points_intensity], axis=-1)

    return points

