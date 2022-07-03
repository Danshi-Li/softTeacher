import copy
import json

from mmdet.datasets import DATASETS, CocoDataset, PIPELINES
from mmdet.datasets.api_wrappers import COCO


@DATASETS.register_module()
class PseudoCocoDataset(CocoDataset):
    def __init__(
        self,
        ann_file,
        pipeline,
        pseudo_ann_file=None,
        caption_ann_file="/home/danshili/softTeacher/data/coco/annotations/captions_train2017.json",
        confidence_threshold=0.9,
        classes=None,
        data_root=None,
        img_prefix="",
        seg_prefix=None,
        proposal_file=None,
        test_mode=False,
        filter_empty_gt=True,
    ):
        self.confidence_threshold = confidence_threshold
        self.caption_ann_file = caption_ann_file
        self.pseudo_ann_file = pseudo_ann_file

        super().__init__(
            ann_file,
            pipeline,
            classes,
            data_root,
            img_prefix,
            seg_prefix,
            proposal_file,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
        )
    
    def load_pesudo_targets(self, pseudo_ann_file):
        with open(pseudo_ann_file) as f:
            pesudo_anns = json.load(f)
        print(f"loading {len(pesudo_anns)} results")

        def _add_attr(dict_terms, **kwargs):
            new_dict = copy.copy(dict_terms)
            new_dict.update(**kwargs)
            return new_dict

        def _compute_area(bbox):
            _, _, w, h = bbox
            return w * h

        pesudo_anns = [
            _add_attr(ann, id=i, area=_compute_area(ann["bbox"]))
            for i, ann in enumerate(pesudo_anns)
            if ann["score"] > self.confidence_threshold
        ]
        print(
            f"With {len(pesudo_anns)} results over threshold {self.confidence_threshold}"
        )

        return pesudo_anns
    
    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.
        Returns:
            list[dict]: Annotation info from COCO api.
        """
        # pesudo_anns = self.load_pesudo_targets(self.pseudo_ann_file)
        self.coco = COCO(ann_file)
        # self.coco.dataset["annotations"] = pesudo_anns
        self.coco.createIndex()

        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info["filename"] = info["file_name"]
            data_infos.append(info)

        with open(self.caption_ann_file,"r") as f:
            self.captions = json.load(f)
        

        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        print(self._parse_ann_info(self.data_infos[idx], ann_info))
        raise ValueError("check get_ann_info internal vars")

        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.
        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.
        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        gt_captions = None

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            captions=gt_captions)

        return ann
