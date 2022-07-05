from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
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
import torchvision.transforms as transforms
import torch.nn.functional as F
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

def reshape_with_padding(img):
    def add_margin(pil_img, top, right, bottom, left, color):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result
    h, w = img.size[1], img.size[0]
    if h > w:
        padding_left = (h - w) // 2
        padding_right = (h - w) - padding_left
        img_padded = add_margin(img,0,padding_right,0,padding_left,(0,0,0))
    else:
        padding_up = (- h + w) // 2
        padding_down = (- h + w) - padding_up
        img_padded = add_margin(img, padding_up,0,padding_down,0,(0,0,0))
    img_resized = img_padded.resize((224,224),resample=Image.BICUBIC)

    return img_resized
        

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',          #6
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',        #11
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',      #17
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',  #24
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',  #30
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',       #35
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',     #39
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',  #46
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',    #52
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',            #58
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',  #64
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',         #69
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',       #75
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',)     #80

#CLASSES = ("cat","dog")
PROMPTS = ("photo of a [CLS]",)

model, preprocess = clip.load("RN50")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

ids = ["350789","581921"]
original_images = []
images = []
texts = []
# plt.figure(figsize=(16, 5))

for id in ids:
    path = "/home/danshili/softTeacher/data/coco/train2017/" + "0"*(12-len(id)) + id + ".jpg"
    #path = "cat_dog.jpg"
    image = Image.open(path).convert("RGB")
    # load original whole pictures
    original_images.append(image)
    preprocessed_image = preprocess(reshape_with_padding(image))
    images.append(preprocessed_image)

for cls in CLASSES:
    texts.append(PROMPTS[0].replace("[CLS]",cls))
image_input = torch.tensor(np.stack(images)).cuda()
text_tokens = clip.tokenize([desc for desc in texts]).cuda()
'''
# CAM class does the forward pass for us. No need to execute it explicitely.
image_features = model.encode_image(image_input).float()
text_features = model.encode_text(text_tokens).float()
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = torch.matmul(text_features,image_features.permute(1,0))
# (text,image) --> (image,text)
# beause CAM takes argmax w.r.t the last axis
similarity = similarity.permute(1,0)
'''
# if take multiple layers as target layer, final output activation map will be averaged over all those layers.
# Note all activation maps are interpolated to that of original image. Can aggregate features of different resolotion.
target_layers = [model.visual.layer4[0]]

cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)

# if take multiple classes as target, will sum gradients of each referred class. 
# TODO: now all classes calculate activation map averaged over all target classes.
# Can we decouple it? Can we get activation map of one image w.r.t one class, ane another image another class?  
#targets = [ClassifierOutputTarget(22),ClassifierOutputTarget(31)]
targets = None # If targets = None, take argmax as target class


cam_input_tensor = (image_input,text_tokens)
grayscale_cam = cam(input_tensor=cam_input_tensor, targets=targets)
#print(grayscale_cam)  # 3-dim lists: [num_images, H, W]


for i in range(len(ids)):
    # bring the channels to the last dimension: (3,224,224) -> (224,224,3)
    visualization_input_img = preprocess_without_normalization(224)(reshape_with_padding(original_images[i])).permute(1,2,0).numpy()
    visualization = show_cam_on_image(visualization_input_img, grayscale_cam[np.newaxis,i])
    cv2.imwrite(f"RN50_extended_cam_cat_overpaint_{ids[i]}.jpg", visualization)
    original_img = show_cam_on_image(visualization_input_img, grayscale_cam[np.newaxis,i], mode="original")
    cv2.imwrite(f"RN50_extended_cam_cat_original_{ids[i]}.jpg", original_img)
    class_specific_img = show_cam_on_image(visualization_input_img, grayscale_cam[np.newaxis,i], mode="product")
    cv2.imwrite(f"RN50_extended_cam_cat_product_{ids[i]}.jpg", class_specific_img)
