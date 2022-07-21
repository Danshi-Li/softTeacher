import torch
import numpy as np
from PIL import Image
import clip
import os
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except:
    BICUBIC = Image.BICUBIC


CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',   #6
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
PROMPTS = ("photo of a [CLS]",)
CLIP = {}
TRAIN_IMG_DIR = '/home/danshili/softTeacher/SoftTeacher/data/coco/train2017/'
VAL_IMG_DIR = '/home/danshili/softTeacher/SoftTeacher/data/coco/val2017/'
SAVE_DIR = '/home/danshili/softTeacher/SoftTeacher/data/gradcam-numpy/'

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def preprocess_without_normalization(n_px):
    return Compose([
        Resize((n_px,n_px), interpolation=BICUBIC),
        #_convert_image_to_rgb,
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
            img_padded = add_margin(img,0,padding_right,0,padding_left,0)
        else:
            padding_up = (- h + w) // 2
            padding_down = (- h + w) - padding_up
            img_padded = add_margin(img, padding_up,0,padding_down,0,0)
        img_resized = img_padded.resize((224,224),resample=Image.BICUBIC)

        return img_resized

def main():
    model, preprocess = clip.load("RN50")
    model.cuda().eval()
    texts = []
    for cls in CLASSES:
        texts.append(PROMPTS[0].replace("[CLS]",cls))
    text_tokens = clip.tokenize([desc for desc in texts]).cuda()
    CLIP['model'] = model
    CLIP['preprocess'] = preprocess
    CLIP['texts'] = text_tokens

    target_layers = [model.visual.layer4[0]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=False)

    #TODO: load all images, train and val, into this list.
    img_list_train = [TRAIN_IMG_DIR + f for f in os.listdir(TRAIN_IMG_DIR)]
    img_list_val = [VAL_IMG_DIR + f for f in os.listdir(VAL_IMG_DIR)]
    img_list_all = img_list_train + img_list_val
    cnt = 0
    for img in img_list_all:
        # iter over all train and val images
        pass
        # 1. first forward pass through CLIP, get classes of interest
        raw_img = Image.open(img)
        img_intformat = reshape_with_padding(raw_img)
        image_input = torch.Tensor(np.stack([CLIP['preprocess'](img_intformat)])).cuda()

        score, _ = CLIP['model'](image_input,CLIP['texts'])
        score_softmax = torch.softmax(score[0],0,score.dtype)
        y=torch.Tensor(range(score_softmax.shape[0]))
        all_cls = y[score_softmax>0.05].int().tolist()
        if len(all_cls) == 0:
            print(img.split("/")[-1].split('.')[0])
            print(score_softmax)

        cam_input_tensor = (image_input,CLIP['texts'])
        activation_map = []
        # 2. second forward pass through CLIP, get activation map of corresponding classes
        for cls in all_cls:
            targets = [ClassifierOutputTarget(int(cls)),]
            grayscale_cam = cam(input_tensor=cam_input_tensor, targets=targets)
            if (np.max(grayscale_cam) > 0.99) or ((cls == all_cls[-1]) and (len(activation_map) == 0)):    # if activation map is not all-zero
                # TODO: for each activation map, resize it back to match the shape of original images.
                #       It would do to centercrop off the padding and then interpolate to original shape.
                #       Also, should visualize to check it indeed matches the original image. (Watch out for Augmentations!)
                #activated_img = show_cam_on_image(preprocess_without_normalization(224)(img_intformat).permute(1,2,0).numpy(),grayscale_cam, mode="product")
                grayscale_cam = np.uint8(grayscale_cam * 255)
                grayscale_cam = np.concatenate([grayscale_cam,grayscale_cam,grayscale_cam])
                grayscale_cam = Image.fromarray(grayscale_cam.transpose((1,2,0)))
                h = int(raw_img.size[1])
                w = int(raw_img.size[0])
                if h > w:
                    activated_img = preprocess_without_normalization(h)(grayscale_cam)
                    #activated_img = Normalize(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0])(activated_img)
                    activated_img = CenterCrop((h,w))(activated_img)
                else:
                    activated_img = preprocess_without_normalization(w)(grayscale_cam)
                    #activated_img = Normalize(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0])(activated_img)
                    activated_img = CenterCrop((h,w))(activated_img)
                
                # 3. save activation map.
                # save activated image to see if painting is correct
                
                activated_img_save = np.float32(activated_img.numpy())[0,:,:]
                activated_img_save = activated_img_save[np.newaxis,:,:]
                #savefile = Image.fromarray(activated_img_save)
                img_num = img.split("/")[-1].split('.')[0]
                np.save(f"{SAVE_DIR}{img_num}_{cls}.npy",activated_img_save)
                #print(f"{SAVE_DIR}{img_num}_{cls}.npy")
        cnt = cnt + 1
        if cnt % 1000 == 0:
            print(cnt)
        


if __name__ == "__main__":
    main()