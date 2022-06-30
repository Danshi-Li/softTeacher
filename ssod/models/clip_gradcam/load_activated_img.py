import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None

import torch
import clip
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except:
    BICUBIC = Image.BICUBIC
    
@PIPELINES.register_module()
class LoadCLIPActivatedImage:
    '''
    specification to be determined
    '''
    def __init__(self):
        '''
        specification to be determined
        '''
        #TODO: these config settings should support reading from config file
        self.CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',   #6
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
        self.PROMPTS = ("photo of a [CLS]",)

        self.clip = {}
        model, preprocess = clip.load("RN50")
        texts = []
        for cls in self.CLASSES:
            texts.append(self.PROMPTS[0].replace("[CLS]",cls))
        text_tokens = clip.tokenize([desc for desc in texts]).cuda()
        self.clip['model'] = model
        self.clip['preprocess'] = preprocess
        self.clip['texts'] = text_tokens

        target_layers = [model.visual.layer4[0]]
        self.cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)


    def __call__(self,results):
        '''
        specification to be determined
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        '''
        
        # Step1: acquire activation maps w.r.t. every classes of interest
        #        (meanwhile deleting activation maps that are too small in magnitude)
        #        The final output is indeterministic in its numbers
        img = self.reshape_with_padding(results['img'])
        image_input = torch.Tensor(img).cuda()
        cam_input_tensor = (image_input,self.clip['text'])

        activation_map = []
        for cls in range(len(self.CLASSES)):
            targets = [ClassifierOutputTarget(cls),]
            grayscale_cam = self.cam(input_tensor=cam_input_tensor, targets=targets)
            if grayscale_cam is not None:    # if activation map output is greater than a threshold magnitude
                activation_map.append(grayscale_cam)

        # step2: For each valid activation maps, compute the activated images and put into pipeline
        img_activated = [torch.Tensor(
                        show_cam_on_image(self.clip['preprocess'](img).permute(1,2,0).numpy(),activation, mode="product")
                        ) for activation in activation_map]
        print(img_activated)
        results["img_activated"] = img_activated
        return results

    def reshape_with_padding(self,img):
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

    def _convert_image_to_rgb(self,image):
        return image.convert("RGB")