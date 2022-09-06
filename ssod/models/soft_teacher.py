import copy
import torch
import random
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models import DETECTORS, build_detector

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.nn.functional as F

import mmcv
import cv2
import numpy as np
from PIL import Image

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None

import clip
from ..gradcam.pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from ..gradcam.pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from ..gradcam.pytorch_grad_cam.utils.image import show_cam_on_image
from .soft_teacher_backup import SoftTeacher


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except:
    BICUBIC = Image.BICUBIC


@DETECTORS.register_module()
class SoftTeacherGradCAM(SoftTeacher):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(SoftTeacherGradCAM, self).__init__(
            model,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )

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
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"]
                ),
                weight=self.unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)

        return loss

    def foward_unsup_train(self, teacher_data, student_data):
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx],
                [teacher_data["proposals"][idx] for idx in tidx]
                if ("proposals" in teacher_data)
                and (teacher_data["proposals"] is not None)
                else None,
                [np.stack(teacher_data["img_metas"][idx]["img_activated"]).transpose(0,3,1,2) for idx in tidx]
                if ("img_activated" in teacher_data["img_metas"][0])
                else None
            )
        student_info = self.extract_student_info(**student_data)

        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def extract_teacher_info(self, img, img_metas, proposals=None, img_activated=None, **kwargs):
        # pad activated image to the size of image, exactly what was done in the collate() function for original images.
        # TODO: check
        #   1. if the padding values are correct
        #   2. if the overall implementation is exact
        if img_activated is not None:
            img_activated_padded = []
            for idx in range(len(img_activated)):
                inner = []
                for instance_idx in range(len(img_activated[idx])):
                    pad = [0 for _ in range(4)]
                    for dim in range(1,2+1):
                        # 'batch_input_shape'=(H,W); img_activated=(3,H,W), do not pad the first dim
                        pad[2*dim-1] = img_metas[0]['batch_input_shape'][-dim] - img_activated[idx][instance_idx].shape[-dim]
                    padded_img = F.pad(torch.Tensor(img_activated[idx][instance_idx]), pad, value=0)
                    inner.append(padded_img)
                img_activated_padded.append(torch.stack(inner).to(img.device))
            img_activated = img_activated_padded
        '''
        print("img")
        print(img.shape)
        print("img_activated")
        t = [img_activated[i].shape for i in range(len(img_activated))]
        print(t)
        '''


        teacher_info = {}
        # there is features from original image
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        # load also features attained from activated
        if img_activated is not None:
            feat_activated = [self.teacher.extract_feat(img) for img in img_activated]
            rpn_activated_out = [list(self.teacher.rpn_head(feat)) for feat in feat_activated]
        else:
            feat_activated = None
            rpn_activated_out = None
        #teacher_info['activated_feature'] = feat_activated
        
        if proposals is None:
            proposal_cfg = self.teacher.train_cfg.get(
                "rpn_proposal", self.teacher.test_cfg.rpn
            )
            rpn_out = list(self.teacher.rpn_head(feat))
            '''
            proposal_list = self.teacher.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
            '''
            proposal_list = self.teacher.rpn_head.get_bboxes(
                *rpn_out, activated_features=rpn_activated_out, img_metas=img_metas, cfg=proposal_cfg
            )
            
        else:
            proposal_list = proposals
        teacher_info["proposals"] = proposal_list

        proposal_list, proposal_label_list = self.teacher.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, self.teacher.test_cfg.rcnn, rescale=False
        )

        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )
        det_bboxes = proposal_list
        reg_unc = self.compute_uncertainty_with_aug(
            feat, img_metas, proposal_list, proposal_label_list
        )
        det_bboxes = [
            torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(det_bboxes, reg_unc)
        ]
        det_labels = proposal_label_list
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        return teacher_info

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
            img_padded = add_margin(img,0,padding_right,0,padding_left,0)
        else:
            padding_up = (- h + w) // 2
            padding_down = (- h + w) - padding_up
            img_padded = add_margin(img, padding_up,0,padding_down,0,0)
        img_resized = img_padded.resize((224,224),resample=Image.BICUBIC)

        return img_resized

    def preprocess_without_normalization(self,n_px):
        return Compose([
            Resize((n_px,n_px), interpolation=BICUBIC),
            self._convert_image_to_rgb,
            ToTensor(),
        ])

    def _convert_image_to_rgb(self,image):
        return image.convert("RGB")



    