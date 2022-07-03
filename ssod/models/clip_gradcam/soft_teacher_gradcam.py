import torch
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models import DETECTORS, build_detector

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid

import mmcv
import numpy as np
from PIL import Image

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None

import clip
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from ..soft_teacher_backup import SoftTeacher


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except:
    BICUBIC = Image.BICUBIC


@DETECTORS.register_module()
class SoftTeacherGradCAM(SoftTeacher):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(SoftTeacherGradCAM, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )

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
        model.cuda().eval()
        texts = []
        for cls in self.CLASSES:
            texts.append(self.PROMPTS[0].replace("[CLS]",cls))
        text_tokens = clip.tokenize([desc for desc in texts])
        self.clip['model'] = model
        self.clip['preprocess'] = preprocess
        self.clip['texts'] = text_tokens

        target_layers = [model.visual.layer4[0]]
        self.cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=False)

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}
        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
            log_every_n(
                {"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            sup_loss = self.student.forward_train(**data_groups["sup"])
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)
        if "unsup_student" in data_groups:
            data_groups['unsup_teacher'] = self.load_gradcam_activated_img(data_groups['unsup_teacher'])
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"]
                ),
                weight=self.unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)

        return loss

    def load_gradcam_activated_img(self, results):

        img = self.reshape_with_padding(results['img'])
        img = img.transpose(2,0,1)
        image_input = torch.Tensor(np.stack([img]))
        cam_input_tensor = (image_input.cuda(),self.clip['texts'].cuda())

        activation_map = []
        for cls in range(len(self.CLASSES)):
            targets = [ClassifierOutputTarget(cls),]
            grayscale_cam = self.cam(input_tensor=cam_input_tensor, targets=targets)
            if np.max(grayscale_cam) > 0:    # if activation map output is greater than a threshold magnitude
                activation_map.append(grayscale_cam)

        # step2: For each valid activation maps, compute the activated images and put into pipeline
        img_activated = [show_cam_on_image(self.clip['preprocess'](img).permute(1,2,0).numpy(),activation, mode="product")
                        for activation in activation_map]
        results["img_activated"] = img_activated
        print(img_activated)
        return results

    def reshape_with_padding(self,img):
        def add_margin(pil_img, top, right, bottom, left, color):
            width, height = pil_img.size
            new_width = width + right + left
            new_height = height + top + bottom
            result = Image.new(pil_img.mode, (new_width, new_height), color)
            result.paste(pil_img, (left, top))
            return result

        img = Image.fromarray(img)
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

        return np.asarray(img_resized)

    def _convert_image_to_rgb(self,image):
        return image.convert("RGB")


        return results

    