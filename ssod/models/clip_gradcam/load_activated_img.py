import mmcv
import numpy as np

from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.datasets.builder import PIPELINES

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None

import torch
import clip
from PIL import Image
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import cv2
import pickle


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
    def __init__(self,keep_ratio=0.01,cls_thr=0.05):
        self.keep_ratio = keep_ratio
        self.cls_thr = cls_thr
        if cls_thr == 0.05:
            self.SAVE_DIR = '/home/danshili/softTeacher/SoftTeacher/data/gradcam-numpy/'
        else:
            self.SAVE_DIR = '/home/danshili/softTeacher/SoftTeacher/data/gradcam-numpy-001/'

    def find_grayscale(self,map,rgb):
        bgr = (rgb[2],rgb[1],rgb[0])
        if bgr in map.keys():
            return map[bgr]
        else:
            # search the nearest neighbor
            bgr = (int(bgr[0]),int(bgr[1]),int(bgr[2]))
            k = 256*256*256
            v = 0
            thr = 4/256/256
            for key,value in map.items():
                diff =np.abs((int(key[0])-bgr[0]) + (int(key[1])-bgr[1])/256 + (int(key[2])-bgr[2])/256/256)
                if diff < thr:
                    return value
                if diff < k:
                    k = diff
                    v = value
            return v

    def __call__(self,results):
        '''
        specification to be determined
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        '''
        
        img_activated = []
        image_number = results['filename'].split('.')[0].split('/')[-1]
        original_img = results['img']


        #TODO: find all activated images corresponding to given image
        for filename in os.listdir(self.SAVE_DIR):
            if image_number in filename:
                if self.cls_thr == 0.05:
                    activation = np.load(self.SAVE_DIR+filename).transpose((1,2,0))
                else:
                    activation = np.load(self.SAVE_DIR+filename)[np.newaxis,:,:].transpose((1,2,0))
                #
                activated_img = activation * activation * original_img * (1 - self.keep_ratio) + original_img * self.keep_ratio
                activated_img = activated_img / np.max(activated_img)
                activated_img = np.uint8(255 * activated_img)
                img_activated.append(activated_img)
                assert activated_img.shape == original_img.shape

        try:
            results["img_activated"] = np.stack(img_activated)
        except:
            raise ValueError(f"UNEXPECTED: image loaded no activation for image number {image_number}")
        return results