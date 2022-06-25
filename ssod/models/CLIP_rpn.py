import statistics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.models import RPNHead
from mmcv.ops import batched_nms
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.anchor_head import AnchorHead
from mmdet.models import HEADS

import clip
import torchvision.transforms as transforms
from torchvision.ops import boxes as box_ops
import json
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
import time

unloader = transforms.ToPILImage()
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

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


@HEADS.register_module()
class CLIPRPNHead(RPNHead):
    def __init__(self,clip_backbone="RN50",**kwargs):
        super(CLIPRPNHead,self).__init__(**kwargs)
        
        model, preprocess = clip.load(clip_backbone)
        model.cuda().eval()
        self.clip = {}
        self.clip["model"] = model
        self.clip["preprocess"] = preprocess

        caption_lst = []
        for cls in CLASSES:
            caption_lst.append(PROMPTS[0].replace("[CLS]",cls))
        self.clip["captions"] = caption_lst

        self.num_classes = len(CLASSES)
        self.tick=time.time()
        self.tock=time.time()
        

    def get_bboxes_with_clip(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   img,
                   cfg=None,
                   gt_bboxes=None,
                   rescale=False):
        
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        statistics = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            image = img[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            gt_bboxes_single = gt_bboxes[img_id]
            proposals, statistic = self._get_bboxes_single_with_clip(cls_score_list, bbox_pred_list,
                                                mlvl_anchors, img_shape,
                                                scale_factor, cfg, image, gt_bboxes_single, rescale)
            statistic["image_id"] = int(img_metas[img_id]["filename"].split("/")[-1].split(".")[0])

            result_list.append(proposals)
            statistics.append(statistic)
        return result_list, statistics

    def _get_bboxes_single_with_clip(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           img,
                           gt_bboxes=None,
                           rescale=False):

        # TODO: before NMS, rank and threshold w.r.t. CLIP similarities
        cfg = self.test_cfg if cfg is None else cfg
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # we set FG labels to [0, num_class-1] and BG label to
                # num_class in other heads since mmdet v2.0, However we
                # keep BG label as 0 and FG label as 1 in rpn head
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            if cfg["nms_pre"] > 0 and scores.shape[0] > cfg["nms_pre"]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)

                topk_inds = rank_inds[:cfg["nms_pre"]]
                scores = ranked_scores[:cfg["nms_pre"]]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg["min_bbox_size"] > 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_inds = torch.nonzero(
                (w >= cfg["min_bbox_size"])
                & (h >= cfg["min_bbox_size"]),
                as_tuple=False).squeeze()
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                ids = ids[valid_inds]

        # TODO: remove the hard coded nms type
        nms_cfg = dict(type='nms', iou_thr=cfg["nms"]["iou_threshold"])
        dets, keep = batched_nms(proposals, scores, ids, nms_cfg)

        self.tick=time.time()
        sim_after_nms = self._calculate_CLIP_similarity_single(img, dets[:,:4], batch_size=256)
        self.tock=time.time()
        elapse = self.tock - self.tick

        print(f"time elapse {elapse}s for similarity calculation, single image")
        #IOU_after_nms = self._calculate_IOU(dets[:,:4],gt_bboxes)
        try:
            IOU_after_nms = bbox_overlaps(gt_bboxes.cpu().numpy(),dets[:,:4].cpu().numpy()).tolist()
        except:
            IOU_after_nms = torch.tensor([]).to(dets.device)
            det_len = dets[:,:4].size()[0]
            gt_len = gt_bboxes.size()[0]
            print(f"UNEXPECTED: IOU matrix has zero length, with proposal len = {det_len} and gt len={gt_len}.")
        statistics = {}
        statistics["sim_after_nms"] = sim_after_nms
        statistics["score_after_nms"] = dets[:,4]
        statistics["IOU_after_nms"] = IOU_after_nms
        statistics["areas_after_nms"] = self.cal_area(dets[:,:4])
        

        return dets[:cfg["max_per_img"]], statistics

    def _calculate_CLIP_similarity_single(self, image, bboxes, batch_size):
        # get CLIP similarity for proposals in a single image
        # if text input does not vary, it is efficient to precalculate at initialization time.
        # it is saved in self.clip["text_features"]
        with torch.no_grad():
            if bboxes.size()[0] == 0:
                print("UNEXPECTED: have zero ground truth bbox!")
                return None

            self.clip["model"].to(image.device)
            image_PIL = tensor_to_PIL(image)
            batch_cnt = (bboxes.size()[0] // batch_size) + (bboxes.size()[0] % batch_size > 0)

            texts = self.clip["captions"]
            text_tokens = clip.tokenize([desc for desc in texts]).to(image.device)
            text_features = self.clip["model"].encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)

            total_similarities = []
            for i in range(batch_cnt):
                #texts = self.clip["captions"]
                if i < (bboxes.size()[0] // batch_size):
                    patchs = [image_PIL
                                .crop(
                                    (float(bbox[0]),
                                    float(bbox[1]),
                                    max(float(bbox[2]),float(bbox[0]+1.0)),
                                    max(float(bbox[3]),float(bbox[1]+1.0)))
                                    ) 
                                for bbox in bboxes[i*batch_size:(i+1)*batch_size,:]]
                else:
                    patchs = [image_PIL
                                .crop(
                                    (float(bbox[0]),
                                    float(bbox[1]),
                                    max(float(bbox[2]),float(bbox[0]+1.0)),
                                    max(float(bbox[3]),float(bbox[1]+1.0)))
                                    ) 
                                for bbox in bboxes[i*batch_size:,:]]

                patchs = [self.clip["preprocess"](patch) for patch in patchs]


                # go through CLIP
                image_input = torch.tensor(np.stack(patchs)).to(image.device)
                #text_tokens = clip.tokenize(["This is " + desc for desc in texts]).to(image.device)

                with torch.no_grad():
                    image_features = self.clip["model"].encode_image(image_input).float()
                    #text_features = self.clip["model"].encode_text(text_tokens).float()

                image_features /= image_features.norm(dim=-1, keepdim=True)
                #text_features /= text_features.norm(dim=-1, keepdim=True)

                # original calculation uses cpu version of matmul, which is not scalable for large ([80*512]x[512*32]) matmuls
                # similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
                similarity = torch.matmul(text_features.to(image.device), image_features.permute(1,0))
                similarity, _ = torch.max(similarity,axis=0)

                total_similarities.append(similarity)
        
        return torch.concat([t for t in total_similarities])


    def _calculate_IOU(self, proposals, gt_bboxes):
        # for each proposal, calculate its IOU with the maximally overlapped gt bbox.
        match_quality_matrix = box_ops.box_iou(gt_bboxes, proposals)
        try:
            IOU_list, _ = match_quality_matrix.max(axis=0)
        except:
            gt_len = gt_bboxes.size()[0]
            proposal_len = proposals.size()[0]
            print(f"UNEXPECTED:inside '_calculate_IOU' at CLIPRPNHead -- got null IOU matrix. gt_bboxes length={gt_len},proposals length={proposal_len}")
            return []

        return IOU_list

    def cal_area(self, proposals):
        return [float((proposal[2]-proposal[0])*(proposal[3]-proposal[1])) for proposal in proposals]