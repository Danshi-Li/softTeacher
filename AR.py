from collections.abc import Sequence
from curses import meta

import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
import torch
import json
import os

AR_thrs = np.arange(0.5,0.95,0.05)
AP_thrs = np.arange(0.5,0.95,0.05)
proposal_nums = [100,200,400,600,800,1200,1600,2400,3200,4800]
ids = []
for root, dirs, files in os.walk("/home/danshili/softTeacher/SoftTeacher/stats/stats-initial/"):
    for f in files:
        ids.append(f.split(".")[0])

def set_recall_param(proposal_nums, iou_thrs):
    """Check proposal_nums and iou_thrs and set correct format.
    """
    if isinstance(proposal_nums, Sequence):
        _proposal_nums = np.array(proposal_nums)
    elif isinstance(proposal_nums, int):
        _proposal_nums = np.array([proposal_nums])
    else:
        _proposal_nums = proposal_nums

    if iou_thrs is None:
        _iou_thrs = np.array([0.5])
    elif isinstance(iou_thrs, Sequence):
        _iou_thrs = np.array(iou_thrs)
    elif isinstance(iou_thrs, float):
        _iou_thrs = np.array([iou_thrs])
    else:
        _iou_thrs = iou_thrs

    return _proposal_nums, _iou_thrs

def _recalls(all_ious, proposal_nums, thrs):

    # img_num = all_ious.shape[0]
    img_num = len(all_ious)
    total_gt_num = sum([ious.shape[0] for ious in all_ious])
    _ious = np.zeros((proposal_nums.size, total_gt_num), dtype=np.float32)
    for k, proposal_num in enumerate(proposal_nums):
        tmp_ious = np.zeros(0)
        for i in range(img_num):
            ious = all_ious[i][:, :proposal_num].copy()
            gt_ious = np.zeros((ious.shape[0]))
            if ious.size == 0:
                tmp_ious = np.hstack((tmp_ious, gt_ious))
                continue
            for j in range(ious.shape[0]):
                gt_max_overlaps = ious.argmax(axis=1)
                max_ious = ious[np.arange(0, ious.shape[0]), gt_max_overlaps]
                gt_idx = max_ious.argmax()
                gt_ious[j] = max_ious[gt_idx]
                box_idx = gt_max_overlaps[gt_idx]
                ious[gt_idx, :] = -1
                ious[:, box_idx] = -1
            tmp_ious = np.hstack((tmp_ious, gt_ious))
        _ious[k, :] = tmp_ious

    _ious = np.fliplr(np.sort(_ious, axis=1))
    recalls = np.zeros((proposal_nums.size, thrs.size))
    for i, thr in enumerate(thrs):
        recalls[:, i] = (_ious >= thr).sum(axis=1) / float(total_gt_num)

    return recalls

def tpfp(all_ious, all_scores, proposal_nums, iou_thr, area_ranges=None):
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_dets = sum([proposal_nums for ious in all_ious])
    num_gts = sum([ious.shape[0] for ious in all_ious])
    tp = np.zeros((1, num_dets), dtype=np.float32)
    fp = np.zeros((1, num_dets), dtype=np.float32)

    tps = []
    fps = []
    for ious, scores in zip(all_ious,all_scores):
        ious_max = ious.max(axis=0)
        ious_argmax = ious.argmax(axis=0)
        sort_inds = np.argsort(-scores)

        for k, (min_area, max_area) in enumerate(area_ranges):
            gt_covered = np.zeros(num_gts, dtype=bool)
            for i in sort_inds[:proposal_nums]:
                if ious_max[i] >= iou_thr:
                    matched_gt = ious_argmax[i]
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                    # otherwise ignore this detected bbox, tp = 0, fp = 0
                elif min_area is None:
                    fp[k, i] = 1
                else:
                    '''
                    bbox = det_bboxes[i, :4]
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area >= min_area and area < max_area:
                        fp[k, i] = 1
                    '''
                    fp[k, i] = 1
        tps.append(tp)
        fps.append(fp)
    return tps, fps

def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).
    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]
    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap





all_ious = []
all_scores = []
all_sims = []
all_ious_sorted_wrt_clip = []
for num in ids:
    with open(f"/home/danshili/softTeacher/SoftTeacher/stats/stats-initial/{num}.json",'r') as f:
        stat = json.load(f)
        if torch.tensor(stat["IOU_after_nms"]).size()[0] == 0:
            print(f"for image NO.{num}, the IOU matrix is null")
            continue
        all_ious.append(np.array(stat["IOU_after_nms"]))
        all_scores.append(np.array(stat["score_after_nms"]))
        all_sims.append(np.array(stat["sim_after_nms"]))

        sims = torch.tensor(stat["sim_after_nms"])
        ranked_scores, rank_inds = sims.sort(descending=True)
        ious_clip_sorted = np.array(stat["IOU_after_nms"])[:,rank_inds]
        all_ious_sorted_wrt_clip.append(ious_clip_sorted)

# calculatr AR

_proposal_nums, _thrs = set_recall_param(proposal_nums,AR_thrs)
recall_score = _recalls(all_ious,_proposal_nums,_thrs)
recall_clip = _recalls(all_ious_sorted_wrt_clip,_proposal_nums,_thrs)
print("recall_scpre")
print(recall_score)
print("recall_clip")
print(recall_clip)


# calculate AP
precisions_score = []
for num in proposal_nums:
    inner = []
    for thr in AP_thrs:
        tps, fps = tpfp(all_ious, all_scores, num, thr)
        tp = sum([t.sum() for t in tps])
        fp = sum([f.sum() for f in fps])
        precision = tp / (tp + fp)
        inner.append(precision)
    precisions_score.append(inner)
print("precisions_score")   
print(precisions_score)

precisions_clip = []
for num in proposal_nums:
    inner = []
    for thr in AP_thrs:
        tps, fps = tpfp(all_ious, all_sims, num, thr)
        tp = sum([t.sum() for t in tps])
        fp = sum([f.sum() for f in fps])
        precision = tp / (tp + fp)
        inner.append(precision)
    precisions_clip.append(inner)
print("precisions_clip")
print(precisions_clip)



# calculate AP
AP_score = average_precision(np.array(recall_score),np.array(precisions_score))
print("AP_score")
print(AP_score)
AP_clip = average_precision(np.array(recall_clip),np.array(precisions_clip))
print("AP_clip")
print(AP_clip)

# save metadata
'''
metadata={}
metadata['recall_score'] = recall_score
metadata["recall_clip"] = recall_clip
metadata["precision_score"] = precisions_score
metadata["precision_clip"] = precisions_clip
metadata["AP_score"] = AP_score
metadata["AP_clip"] = AP_clip

for k,v in metadata.items():
    if (type(v) == np.ndarray) or (type(v) == torch.tensor):
        metadata[k] = v.tolist()

with open(f"/home/danshili/softTeacher/SoftTeacher/stats/metadata/metadata.json",'w') as f:
    f.write(json.dumps(metadata))
'''
