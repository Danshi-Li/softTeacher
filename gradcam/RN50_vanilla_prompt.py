from .pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from .pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .pytorch_grad_cam.utils.image import show_cam_on_image
import clip
import os
#import skimage
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from collections import OrderedDict
import torch
import cv2
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def preprocess_without_normalization(n_px):
    return Compose([
        Resize((n_px,n_px), interpolation=BICUBIC),
        #CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
    ])

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
PROMPTS = ("photo of a [CLS]",)

model, preprocess = clip.load("RN50")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

ids = ["581921"]
original_images = []
images = []
texts = []
# plt.figure(figsize=(16, 5))

for id in ids:
    path = "/home/danshili/softTeacher/data/coco/train2017/" + "0"*(12-len(id)) + id + ".jpg"
    image = Image.open(path).convert("RGB")
    # load original whole pictures
    original_images.append(image)
    preprocessed_image = preprocess(image)
    images.append(preprocessed_image)

for cls in CLASSES:
    texts.append(PROMPTS[0].replace("[CLS]",cls))

image_input = torch.tensor(np.stack(images)).cuda()
text_tokens = clip.tokenize([desc for desc in texts]).cuda()
image_features = model.encode_image(image_input).float()
text_features = model.encode_text(text_tokens).float()
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = torch.matmul(text_features,image_features.permute(1,0))
# (text,image) --> (image,text)
# beause CAM takes argmax w.r.t the last axis
similarity = similarity.permute(1,0)

target_layers = [model.visual.layer4[-1]]

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
targets = None
cam_input_tensor = (image_input,text_tokens)
grayscale_cam = cam(input_tensor=cam_input_tensor, targets=targets)

# bring the channels to the first dimension: (3,224,224) -> (224,224,3)
visualization_input_img = preprocess_without_normalization(224)(image).permute(1,2,0).numpy()
visualization = show_cam_on_image(visualization_input_img, grayscale_cam, use_rgb=True)
cv2.imwrite("RN50_cam_overpaint.jpg", visualization)
original_img = show_cam_on_image(visualization_input_img, grayscale_cam, use_rgb=True, mode="original")
cv2.imwrite("RN50_cam_original.jpg", original_img)
class_specific_img = show_cam_on_image(visualization_input_img, grayscale_cam, use_rgb=True, mode="product")
cv2.imwrite("RN50_cam_product.jpg", class_specific_img)
