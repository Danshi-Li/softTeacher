import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from PIL import Image

def preprocess_image(img: np.ndarray, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      mode = "overpaint",
                      keep_ratio = 0.01) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param keep_ratio: The weight of original image signal kept for activated image input
    :returns: The default image with the cam overlay.
    """
    mask_uint8 = np.uint8(255*mask)[0]
    heatmap = cv2.applyColorMap(mask_uint8, colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    if mode == 'overpaint':
        cam = heatmap + img
    elif mode == "product":
        activation = mask.transpose((1,2,0))
        cam = activation * activation * img * (1 - keep_ratio) + img * keep_ratio
    elif mode == "original":
        cam = img
    elif mode == "heatonly":
        cam = heatmap
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        img = img.astype(float)
        if target_size is not None: 
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result

def scale_accross_batch_and_channels(tensor, target_size):
    batch_size, channel_size = tensor.shape[:2]
    reshaped_tensor = tensor.reshape(
        batch_size * channel_size, *tensor.shape[2:])
    result = scale_cam_image(reshaped_tensor, target_size)
    result = result.reshape(
        batch_size,
        channel_size,
        target_size[1],
        target_size[0])
    return result
