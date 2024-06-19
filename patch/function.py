import sys
sys.path.append("/home/qiuchengyu/facedetection/AdversarialMask") 
import os
import numpy as np
import fnmatch
import glob
import json
from PIL import Image,ImageFile
import torch.nn.functional as F
import torch
import random
import os
import math
#import utils

import torchvision
#from utils.median_pool import MedianPool2d
from tqdm import tqdm
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torchvision import transforms
from collections import OrderedDict
import dlib
import imutils
from imutils import face_utils
from imutils import paths
import numpy as np
from collections import namedtuple
import torch
from torch.utils import model_zoo
import torch.optim as optim
import torch.nn as nn
import cv2
from torchvision import transforms as T
from face_detector import YoloDetector
#import dlib
from models.utils.general import xywh2xyxy
import os 
import PIL
from scipy import integrate
from scipy.optimize import fsolve

import face_recognition.insightface_torch.backbones as InsightFaceResnetBackbone
import face_recognition.magface_torch.backbones as MFBackbone
from landmark_detection.face_alignment.face_alignment import FaceAlignment, LandmarksType
from landmark_detection.pytorch_face_landmark.models import mobilefacenet

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from models.experimental import attempt_load
import os 
import argparse
from pathlib import Path
import numpy as np
from models.utils.general import check_img_size, non_max_suppression_face
from glob import glob

from skimage.transform import resize

from chrislib.general import (
    invert, 
    uninvert, 
    view, 
    np_to_pil, 
    to2np, 
    add_chan, 
    show, 
    round_32,
    tile_imgs
)
from chrislib.data_util import load_image
from chrislib.normal_util import get_omni_normals
import cv2
from boosted_depth.depth_util import create_depth_models, get_depth

from intrinsic.model_util import load_models
from intrinsic.pipeline import run_pipeline

from intrinsic_compositing.shading.pipeline import (
    load_reshading_model,
    compute_reshading,
    generate_shd,
    get_light_coeffs
)

from intrinsic_compositing.albedo.pipeline import (
    load_albedo_harmonizer,
    harmonize_albedo
)

from omnidata_tools.model_util import load_omni_model
class ThermalDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        
        self.images  = list(paths.list_files(self.img_dir))
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        image = cv2.imread(path)
        image = cv2.resize(image,(112,112))
        return image

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
# Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def check_img_size(img_size, s=32):
# Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    '''
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    '''
    return new_size

def make_divisible(x, divisor):
# Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor
    


class CustomDataset1(Dataset):
    def __init__(self, visual_img_dir,thermal_img_dir, img_size,data_detector,mode, shuffle=True, transform=None):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.visual_img_dir = visual_img_dir
        self.thermal_img_dir = thermal_img_dir
        self.target_size = None
        self.img_size = img_size
        self.shuffle = shuffle
        self.data_detector = data_detector
        self.visual_img_names = []#self.get_image_names(indices)
        self.thermal_img_names = []
        self.files = os.walk(self.visual_img_dir)  
        self.thermal_files = os.walk(self.thermal_img_dir)  
        self.mode = mode
        num = 0
        if mode == 'train':
            for path,dir_list,file_list in self.files:  
                for file_name in file_list:  
                    if file_name.endswith('jpg') or file_name.endswith('png') or file_name.endswith('JPG'):
  
                    #if file_name == 'front_face.jpg' :
                        visual_img_path = os.path.join(path,file_name)
                        thmal_fine_name = file_name[:-5] + '1' + file_name[-4:]
                        thermal_img_path = os.path.join(thermal_img_dir,thmal_fine_name)
                        if os.path.isfile(thermal_img_path):
                            self.visual_img_names.append(visual_img_path)
                            self.thermal_img_names.append(thermal_img_path)
                        else:
                            continue
                    '''if len(self.visual_img_names) == 500:
                        break
                if len(self.visual_img_names) == 500:
                    break'''
        else:
            for path,dir_list,file_list in self.thermal_files:  
                for file_name in file_list:  
                    if file_name.endswith('jpg') or file_name.endswith('png') or file_name.endswith('JPG'):
                        thermal_img_path = os.path.join(path,file_name)
                        self.thermal_img_names.append(thermal_img_path)
                        
            for path,dir_list,file_list in self.files:  
                for file_name in file_list:  
                    if file_name.endswith('jpg') or file_name.endswith('png') or file_name.endswith('JPG'):
                        visual_img_path = os.path.join(path,file_name)
                        self.visual_img_names.append(visual_img_path)
                        
        
        
        self.transform = transform

    def __len__(self):
        if len(self.visual_img_names) < len(self.thermal_img_names):
            return len(self.visual_img_names)
        else:
            return len(self.thermal_img_names)
    def __getitem__(self, idx):
        
        assert idx <= len(self), 'index range error'
        visual_img_path = self.visual_img_names[idx]

        print('visual_img_path=',visual_img_path)
        img = Image.open(visual_img_path)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = np.array(img)
        
 
        #img = cv2.resize(img,(128,128))
        
        #img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
        if self.mode == 'train':
            img = cv2.resize(img,self.img_size)

        
        img = img[:,:,:3]
  
        cv2.imwrite('processing_images/origin_img11.jpg',img[:,:,::-1])
        if img.ndim==2:
            img = np.expand_dims(img,2)
        
        if img.shape[2] == 1:
            img = np.repeat(img,repeats=3,axis=2)
            
        height = img.shape[0]
        width = img.shape[1]
        
        
        h0, w0 = img.shape[:2]  # orig hw
        max_size = 0
        num = 0
        total_size = []
        if h0 > w0:
            max_size = h0
        else:
            max_size = w0
        test_size_image = np.zeros_like(img)
        
        while max_size> 30:
            if self.target_size:
                r = self.target_size / min(h0, w0)  # resize image to img_size
                if r < 1:  
                    test_size_image = cv2.resize(test_size_image, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
            
            imgsz = check_img_size(max(test_size_image.shape[:2]), s=self.data_detector.detector.module.stride.max())  # check img_size
            new_img = letterbox(test_size_image, new_shape=imgsz)[0]
            
            h1 , w1 = new_img.shape[:2]
            original_size = [h0,w0]
            if h1 < w1:
                new_size = [h1,h1]
            else:
                new_size = [w1,w1]
            total_size.append(new_size)
            h0 = int(h0/2)
            w0 = int(w0/2)
            test_size_image = cv2.resize(test_size_image,(w0,h0))
            max_size = max_size/2
            num = num + 1
        #total_size = total_size[::-1]
        
        img = img.transpose(2, 0, 1).copy()
        img = torch.from_numpy(img)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        #img = torchvision.transforms.functional.rotate(img, -90)
        
        
        thermal_img_path = self.thermal_img_names[idx]
     
        #img_path = 'datasets/thermal/gray/train/images/17_2_2_2_1107_45_1.png'
        #img_path = 'physical_images/examole_infrared.jpg'
        ###rgb

       
        ###thermal
        #img_path = 'datasets/image4.jpg'
  
     
        thermal_img = Image.open(thermal_img_path)#.convert('RGB')
        
        thermal_img = self.transform(thermal_img)
 
        thermal_img = thermal_img[:3,:,:]
        
        return img,thermal_img
    
    def get_image_names(self, indices):
        files_in_folder = get_nested_dataset_files(self.img_dir, self.celeb_lab_mapper.keys())
        files_in_folder = [item for sublist in files_in_folder for item in sublist]
        if indices is not None:
            files_in_folder = [files_in_folder[i] for i in indices]
        png_images = fnmatch.filter(files_in_folder, '*.png')
        jpg_images = fnmatch.filter(files_in_folder, '*.jpg')
        jpeg_images = fnmatch.filter(files_in_folder, '*.jpeg')
        return png_images + jpg_images + jpeg_images





from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt




def transparent_back(pic_path):    # make jpg picture's background transparent(png)
    img = cv2.imread(pic_path)     # array
    sticker = Image.open(pic_path) # image
    W,H = sticker.size
    #print(W,H)
    mask = np.zeros(img.shape[:2],np.uint8)
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (1,1,450,450)
    
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT) 
    #print(mask)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8').transpose()
    print('mask2 = ',mask2.shape)

    #print(mask2[200][200])
    sticker = sticker.convert('RGBA')
    for i in range(W):
        for j in range(H):
            color_1 = sticker.getpixel((i,j))
            if(mask2[i][j]==0):   # transparent
                color_1 = color_1[:-1] + (0,)
                sticker.putpixel((i,j),color_1)
            else:
                color_1 = color_1[:-1] + (255,)
                sticker.putpixel((i,j),color_1)
    sticker.show()
    sticker.save(pic_path[:-3]+'png')

def make_stick2(backimg,sticker,x,y,factor=1):
    
    backimg = np.array(backimg)
    backimg = backimg[:,:,:3]
    r,g,b = cv2.split(backimg)
    background = cv2.merge([b, g, r])
    #print('background = ',background.shape)
    
    base,_ = make_basemap(background.shape[1],background.shape[0],sticker,x=x,y=y)
    #print('basemap = ',basemap.shape)
    #print('basemap = ',basemap[100][130][3])
    r,g,b,a = cv2.split(base)
    foreGroundImage = cv2.merge([b, g, r,a])
    # cv2.imshow("outImg",foreGroundImage)
    # cv2.waitKey(0)

    b,g,r,a = cv2.split(foreGroundImage)
    foreground = cv2.merge((b,g,r))
    
    alpha = cv2.merge((a,a,a))

    foreground = foreground.astype(float)
    background = background.astype(float)
    
    alpha = alpha.astype(float)/255
    alpha = alpha * factor
    #print('alpha = ',alpha)
    
    foreground = cv2.multiply(alpha,foreground)
    background = cv2.multiply(1-alpha,background)
    
    outarray = foreground + background
    #cv2.imwrite("outImage.jpg",outImage)

    # cv2.imshow("outImg",outImage/255)
    # cv2.waitKey(0)
    b, g, r = cv2.split(outarray)
    outarray = cv2.merge([r, g, b])
    outImage = Image.fromarray(np.uint8(outarray))
    return outImage


def img_to_cv(image):
    imgarray = np.array(image)
    r,g,b,a = cv2.split(imgarray)
    cvarray = cv2.merge([b, g, r, a])
    return cvarray

def change_sticker(sticker,scale):
    #sticker = Image.open(stickerpath)
    new_weight = int(sticker.size[0]/scale)
    new_height = int(sticker.size[1]/scale)
    #print(new_weight,new_height)
    sticker = sticker.resize((new_weight,new_height),Image.LANCZOS)
    return sticker

def make_basemap(width,height,sticker,x,y):
    layer = Image.new('RGBA',(width,height),(255,255,255,0)) # white and transparent
    
    layer.paste(sticker,(x,y))
    #layer.show()
    base = np.array(layer)
    alpha_matrix = base[:,:,3]
    basemap = np.where(alpha_matrix!=0,1,0)
    return base,basemap










@torch.no_grad()
def apply_mask(location_extractor, fxz_projector, img_batch, patch_rgb, patch_alpha=None, is_3d=False):
    preds = location_extractor(img_batch)
    img_batch_applied = fxz_projector(img_batch, preds, patch_rgb, uv_mask_src=patch_alpha, is_3d=is_3d)
    return img_batch_applied


@torch.no_grad()
def load_mask(config, mask_path, device):
    transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])
    img = Image.open(mask_path)
    img_t = transform(img).unsqueeze(0).to(device)
    
        
        
        
    if config.mode=='test':
        mask_b = Image.open('data/masks/blue.png')
        uv_face = transform(mask_b).unsqueeze(0).to(device)
        uv_face = uv_face[:,3].unsqueeze(0)
        uv_face[uv_face<0.5] = 0
        uv_face[uv_face>0.5] = 1

        img_t = torch.cat([img_t,uv_face],1)
    
    #img_t = img_t #* uv_face
    return img_t


def get_landmark_detector(config, device):
    landmark_detector_type = config.landmark_detector_type
    if landmark_detector_type == 'face_alignment':
        return FaceAlignment(LandmarksType._2D, device=str(device))
    elif landmark_detector_type == 'mobilefacenet':
        model = mobilefacenet.MobileFaceNet([112, 112], 136).eval().to(device)
        sd = torch.load('landmark_detection/pytorch_face_landmark/weights/mobilefacenet_model_best.pth.tar', map_location=device)['state_dict']
        model.load_state_dict(sd)
        return model
    
def get_thermal_predictor(device):
    print("[INFO] loading dlib thermal face detector...")
    detector = dlib.simple_object_detector(os.path.join('models', "dlib_face_detector.svm"))
    
    # load the facial landmarks predictor
    print("[INFO] loading facial landmark predictor...")
    predictor = dlib.shape_predictor(os.path.join('models', "dlib_landmark_predictor.dat"))
    return detector, predictor

def get_nested_dataset_files(img_dir, person_labs):
    files_in_folder = [glob.glob(os.path.join(img_dir, lab, '**/*.*g'), recursive=True) for lab in person_labs]
    
    return files_in_folder


def get_split_indices(img_dir, celeb_lab, num_of_images):
    dataset_nested_files = get_nested_dataset_files(img_dir, celeb_lab)
  
    nested_indices = [np.array(range(len(arr))) for i, arr in enumerate(dataset_nested_files)]
    
    nested_indices_continuous = [nested_indices[0]]
    
    for i, arr in enumerate(nested_indices[1:]):
        nested_indices_continuous.append(arr + nested_indices_continuous[i][-1] + 1)
    train_indices = np.array([np.random.choice(arr_idx, size=num_of_images, replace=False) for arr_idx in
                        nested_indices_continuous]).ravel()
    test_indices = list(set(list(range(nested_indices_continuous[-1][-1]))) - set(train_indices))

    return train_indices, test_indices


def get_train_loaders(config,data_detector):
    train_dataset = CustomDataset1(visual_img_dir=config.visual_img_dir,
                                   thermal_img_dir=config.thermal_img_dir,
                                   img_size=config.img_size,
                                   mode = config.mode,
                                   data_detector = data_detector,
                                   transform=transforms.Compose(
                                       [#transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
                                        transforms.Resize(config.img_size),
                                        transforms.ToTensor()]))
   
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size,shuffle=False)

    return train_loader



def get_patch(config):
    if config.initial_patch == 'random':
        patch = torch.rand((1, 3, config.patch_size[0], config.patch_size[1]), dtype=torch.float32)
    elif config.initial_patch == 'white':
        patch = torch.ones((1, 3, config.patch_size[0], config.patch_size[1]), dtype=torch.float32)
    elif config.initial_patch == 'black':
        patch = torch.zeros((1, 3, config.patch_size[0], config.patch_size[1]), dtype=torch.float32) +0.01
    uv_face = transforms.ToTensor()(Image.open('prnet/new_uv.png').convert('L'))
    #patch = gaussian_filter(patch)
    patch = patch * uv_face
    patch.requires_grad_(True)
    
    return patch

def get_test_patch(config):
    patch = torch.zeros((1, 3, config.patch_size[0], config.patch_size[1]), dtype=torch.float32) + 0.1
    uv_face = transforms.ToTensor()(Image.open('prnet/new_uv.png').convert('L'))
    patch = patch * uv_face
    patch.requires_grad_(True)
    
    return patch

    

def diff_evo(patches,best_loss,img_batch,forward_step,nps_calculator,best_patch,device):

    Rm = 0.5
    Rc = 0.6
    best_score_diff = 0
    new_patches = patches
    
    step = 20
   # nest_generation = patches
    for i in range(step):
        print('the step:',i)
        selected_block = []
        NP = len(patches)
        #print('all_blocks_shape=',np.array(all_blocks).shape)
        
        #变异操作，和遗传算法的变异不同！,得到任意两个个体的差值，与变异算子相乘加第三个个体
       
        for m in range(NP):      
            r1=m
            r2=random.sample(range(0,NP),1)[0]
            while r2==m or r2==r1:
                r2=random.sample(range(0,NP),1)[0]
            r3=random.sample(range(0,NP),1)[0]
            while r3==m or r3==r2 or r3==r1:
                r3=random.sample(range(0,NP),1)[0]
            
            #print('block[r2]=',Blocks[r2])
            #print('block[r3]=',Blocks[r3])
            
            Parents = patches[r1]
            
            patches[r1] = patches[r1]+Rm*(patches[r2]-patches[r3])
            
            #print('blocks=',Blocks)
            #print('len=',len(block_mask))
            #print('box=',box)
            
            #patches[r1][patches[r1]>1] = 1
            #patches[r1][patches[r1]<0] = 0
                
            patches[r1].data.clamp_(0, 1)
            
            #vec = (Blocks[r1]+Rm*(Blocks[r2]-Blocks[r3]))
            
            #print('len1=',len(Blocks))
            
            Children = patches[r1]
           
            
            
            '''rand = random.random()
            if rand <= Rc:
                Children = Parents'''
            
            children_det_loss,tv_loss = forward_step(img_batch,Children)
            if Children.shape[1]==4:
                nps = nps_calculator(Children[:,:3].to(device))
            else:
                nps = nps_calculator(Children.to(device))
          
            nps = nps  *0.01
            children_loss = children_det_loss  + nps+tv_loss
            

            
            
            
            det_loss,tv_loss = forward_step(img_batch,Parents)
            if Parents.shape[1]==4:
                nps = nps_calculator(Parents[:,:3].to(device))
            else:
                nps = nps_calculator(Parents.to(device))
            
            nps = nps *0.01
            parents_loss = det_loss  + nps +tv_loss
            
          
            #print('best_loss_previous=',best_loss)
            if children_loss >= parents_loss: 
                patches[r1] = Parents
                if parents_loss < best_loss:
                    best_patch = Parents
                    best_loss = parents_loss
                
            elif children_loss < parents_loss:
                patches[r1] = Children
                if children_loss < best_loss:
                    best_patch = Children
                    best_loss = children_loss
            
        
        
    return best_patch, patches


def gaussian_filter(img, K_size=3, sigma=1.3):
    
    if len(img.shape) == 3:
        img = img.permute(1,2,0)
        H, W, C = img.shape
    elif len(img.shape) == 4:
        img = img.squeeze(0).permute(1,2,0)
        H, W, C = img.shape
 
    ## Zero padding
    pad = K_size // 2
    out = torch.zeros((H + pad * 2, W + pad * 2, C)).float()
    out[pad: pad + H, pad: pad + W] = img.float()
    ## prepare Kernel
    K = torch.zeros((K_size, K_size), dtype=torch.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * torch.pi * sigma * sigma)
    K /= K.sum()
    tmp = out
    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                if out[pad + y, pad + x, 0] == 0 and out[pad + y, pad + x, 1] == 0 and out[pad + y, pad + x, 2] == 0:
                    break
                out[pad + y, pad + x, c] = torch.sum(K * tmp[y: y + K_size, x: x + K_size, c])
    out = torch.clip(out, 0, 1)
    out = out[pad: pad + H, pad: pad + W]
    out = out.permute(2,0,1).unsqueeze(0)
    
    return out


def p_selection(it,p_init,n_query):
    if 10/10000*n_query < it <= 50/10000*n_query:
        p = p_init / 2
    elif 50/10000*n_query < it <= 200/10000*n_query:
        p = p_init / 4
    elif 200/10000*n_query < it <= 500/10000*n_query:
        p = p_init / 8
    elif 500/10000*n_query < it <= 1000/10000*n_query:
        p = p_init / 16
    elif 1000/10000*n_query < it <= 2000/10000*n_query:
        p = p_init / 32
    elif 2000/10000*n_query < it <= 4000/10000*n_query:
        p = p_init / 64
    elif 4000/10000*n_query < it <= 6000/10000*n_query:
        p = p_init / 128
    elif 6000/10000*n_query < it <= 8000/10000*n_query:
        p = p_init / 256
    elif 8000/10000*n_query < it <= 10000*n_query:
        p = p_init / 512
    else:
        p = p_init
    return p

def random_int(low=0, high=1, shape=[1],device=None):
        t = low + (high - low) * torch.rand(shape).to(device)
        return t.long()
    
def random_choice(shape,device):
        t = 2 * torch.rand(shape).to(device) - 1
        m = torch.sign(t)
        
        return torch.sign(t)
    
    
def random_search(forward_step,img_batch,patches,n_queries,iterat,p_init,image_size,applied_mask,best_loss,device,threshold,do_aug):
    
    loss_list = []
    loss_ind = []

    
    img_batch = F.interpolate(img_batch,(image_size[0],image_size[1]))
    patches = F.interpolate(patches,(image_size[0],image_size[1]))
    applied_mask = F.interpolate(applied_mask,(image_size[0],image_size[1]))
    loss_init,applied_mask,applied_image,applied_patch = forward_step(img_batch,patches,image_size,first=False)
    print('loss_init=',loss_init,'iterat=',iterat,'n_queries=',n_queries)
    if loss_init < threshold:
        return patches,loss_init,-1
    
    mask_find = applied_mask[0,0]
    save_mask = applied_mask[0].cpu().numpy().transpose(1,2,0)
    
    save_image = img_batch[0].cpu().numpy().transpose(1,2,0)
    
    #cv2.imwrite('save_patches/mask_416.bmp',save_mask*255)
    #cv2.imwrite('save_patches/image_416.bmp',save_image[:,:,::-1]*255)
    coor = torch.nonzero(mask_find)
    y_coor = coor[:,0]
    x_coor = coor[:,1]

    x_min = x_coor.min().item()
    x_max = x_coor.max().item()
    y_min = y_coor.min().item()
    y_max = y_coor.max().item()
    loss_min = best_loss
    '''if loss_min < threshold:
        return patches,loss_min'''
    total_it = iterat
    for it in range(iterat, n_queries):
        total_it = it
        b,c,h,w = img_batch.shape
        s_h = y_max - y_min
        s_w = x_max - x_min
        if s_h>s_w:
            s_w = s_h
        else:
            s_h = s_w
        
        
        s_h_it = int(max(p_selection(it,p_init,n_queries) ** .5 * s_h, 1))
        s_w_it = int(max(p_selection(it,p_init,n_queries) ** .5 * s_w, 1))
        p_it_height = torch.randint(y_min,y_max-s_h_it,size=(1,)) 
        p_it_width = torch.randint(x_min,x_max-s_w_it,size=(1,))
        # sample update
        '''margin = 30
        s = 112 - 2*margin
        #s = 112
        s_it = int(max(p_selection(it,p_init) ** .5 * s, 1))
        p_it = torch.randint(s - s_it + 1, size=[2])
        p_it_height = torch.randint(margin,(s - s_it + 1+margin),size=(1,)) 
        p_it_width = torch.randint(0,(s+2*margin - s_it + 1),size=(1,))'''
        # 30 --> 92

        patches_new = patches.clone()
        img_batch_new = img_batch.clone()
        color_store = load_color('non_printability/30values.txt')
        #it_start_cu = it + (n_queries - it) // 2
        for counter in range(img_batch_new.shape[0]):
                
                if it < 0.9*n_queries:
                    if s_h_it > 1:
                        color_ind = random.randint(0,29)
                        color = color_store[color_ind]
                        red, green, blue = color[0], color[1], color[2]
                        patches_new[counter,0,p_it_height:p_it_height + s_h_it, p_it_width:p_it_width + s_w_it] = red
                        patches_new[counter,1,p_it_height:p_it_height + s_h_it, p_it_width:p_it_width + s_w_it] = green
                        patches_new[counter,2,p_it_height:p_it_height + s_h_it, p_it_width:p_it_width + s_w_it] = blue
                        #patches_new[counter, :3, p_it_height:p_it_height + s_it, p_it_width:p_it_width + s_it] += random_choice([c, 1, 1],device)
                        
                    #patches_new[counter, :3, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it] += random_choice([c, 1, 1],device)
                    else:
                        # make sure to sample a different color
                        old_clr = patches_new[counter, :3, p_it_height:p_it_height + s_h_it,p_it_width:p_it_width + s_w_it].clone().to(device)
                        #old_clr = patches_new[counter, :3, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it].clone()
                        new_clr = old_clr.clone().to(device)
                        while (new_clr.cpu().equal(old_clr.cpu())):
                            #new_clr = torch.rand([c, 1, 1]).clone().clamp(0., 1.)
                            color_ind = random.randint(0,29)
                            color = color_store[color_ind]
                            red, green, blue = color[0], color[1], color[2]
                            new_clr[0,:, :] = red
                            new_clr[1,:, :] = green
                            new_clr[2,:, :] = blue
                        patches_new[counter, :3, p_it_height:p_it_height + s_h_it, p_it_width:p_it_width + s_w_it] = new_clr.clone()
                        #patches_new[counter, :3, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it] = new_clr.clone()
                else:
                    
                    assert it >= 0.9*n_queries
                    # single channel updates
                    new_ch = random_int(low=0, high=3, shape=[1],device=device)
                    patches_new[counter, new_ch, p_it_height:p_it_height + s_h_it, p_it_width:p_it_width + s_w_it] = 1. - patches_new[
                        counter, new_ch, p_it_height:p_it_height + s_h_it, p_it_width:p_it_width + s_w_it]
                    '''patches_new[counter, new_ch, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it] = 1. - patches_new[
                                    counter, new_ch, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it]'''
                    
                #patches_new[counter].clamp_(0.1, 0.9)
        loss,applied_mask,applied_image,applied_patch = forward_step(img_batch,patches_new,image_size = image_size,first=False)
        
        
        if loss<= loss_min:
            loss_min = loss
            patches = patches_new
            
            mask = applied_mask.squeeze(0).permute(1,2,0).cpu().numpy()
            save_patch = F.interpolate(patches_new,(applied_image.shape[-2],applied_image.shape[-1]))
            save_patch = save_patch.squeeze(0).cpu().detach().numpy().transpose(1,2,0)
            #save_patch_mask = np.expand_dims(save_patch[:,:,3],axis=2)
            save_patch = save_patch[:,:,:3]
            cv2.imwrite('save_patches/save_patch_mogface.jpg',save_patch[:,:,::-1]*mask*255)


            '''save_image = applied_image.squeeze(0).cpu().detach().numpy().transpose(1,2,0)
            cv2.imwrite('processing_images/best_image.jpg',save_image[:,:,::-1]*255)'''
        
        #loss_min[idx_to_update] = loss[idx_to_update]
        total_loss = loss.item()

        
        print('the iteration:%d,image size:%d, min loss:%f,total loss:%f'
            %(it,image_size[0],loss_min,total_loss))
        
        if loss_min < threshold:
            break
    
    
        
    print('final result: image size:%d,min loss:%f,total iteration:%d'%(image_size[0],loss_min,total_it))
    return patches, loss_min,total_it






def load_color(printability_file):
    printability_list = []
    with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

    printability_array = []
    for printability_triplet in printability_list:
        printability_imgs = []
        red, green, blue = printability_triplet
        printability_imgs.append(red)
        printability_imgs.append(green)
        printability_imgs.append(blue)
        printability_array.append(printability_imgs)

    printability_array = np.asarray(printability_array)
    printability_array = np.float32(printability_array)
    pa = torch.from_numpy(printability_array)
    return pa


def patch_optimization(forward_step,forward_step_origin,img_batch,patches,n_queries,image_size,device,threshold,do_aug):
    loss_list = []
    loss_ind = []
    #loss = torch.tensor(loss)
    img_batch = F.interpolate(img_batch,(image_size[0],image_size[1]))
    patches = F.interpolate(patches,(image_size[0],image_size[1]))
    
    '''loss_init,applied_mask,applied_image,applied_patch = forward_step(img_batch,patches,image_size,first=False)
    if loss_init < 0.8:
        return patches,loss_init,-1'''
    
    '''mask = applied_mask[0,0]
    mask = torch.unsqueeze(mask, axis=0).repeat(3, 1, 1).unsqueeze(0)'''
    if patches.shape[1] == 4:
        patches = patches[:,:3,:,:]
    
    
    fmax = float('inf')
    pattern = patches
    lr = 0.3
    eps = 1e-10
    grad_momentum = 0
    full_matrix = 0
    beta1 = 0.5
    beta2 = 0.5
    noise_number = 30
    signs = None
    favg_diff = 0
    last_grad = 0
    #last_loss = loss_init
    best_loss = 1
    best_loss_iter = 0
    count = 0
    #hist_favg = [(1-loss_init)]
    hist_pattern = [patches]
    
    
  
    noise_shape = [100] + list(img_batch[0].shape)   

    #noise_shape = img_batch.repeat(noise_number,img_batch.shape[1],img_batch.shape[2],img_batch.shape[3]).shape
    input_size = img_batch.shape[2]
    search_iter = n_queries
    nn_graph = np.zeros((search_iter+1, search_iter+1))
    loss_list = []

    last_perturbed = patches


    stage1_total = int(n_queries * 0.1)  #1000
    stage1_iter  = int(stage1_total / 2) #500
    stage1_trial = 2

    box, size = box_determine(img_batch,device=device)
    print('size=',size)
    x_min,y_min,x_max,y_max = box[0],box[1],box[2],box[3]
    input_size_box = int((y_max-y_min)/2)

    '''xi = (x_min)+int(np.ceil((input_size_box - size) / 2 + 1))
    yi = y_min + int((y_max - y_min)/2) + int(np.ceil((input_size_box - size) / 2 + 1))'''
    x_mid = int(x_min + (x_max - x_min)/2)
    y_mid = int(y_min + (y_max - y_min)/2)
    '''side = input_size_box // 5
    region = True
    print('-'*50)
    print('Initial search...')
    print('-'*50)
    
    for i in range(stage1_iter):
        ### initialize location ###
        mask = torch.rand(img_batch[0, 0].shape)

        if i == 0:
            xi, yi = best_xi, best_yi
        else:
            stride = 1 if torch.rand(1)[0] < 0.5 else 2
            if region:
                stride *= 4

            remain = input_size_box - size + 1 - side * 2

            
            
            xi = torch.randint(0,1, (1,))[0] * stride + side
            yi = torch.randint(0,1, (1,))[0] * stride + side

        if i > 10:
            local = 2
            if region:
                local = 8
            xi = best_xi + np.random.randint(-local, local)
            yi = best_yi + np.random.randint(-local, local)
'''
        ### initialize mask ###
    mask = torch.zeros(img_batch[0, 0].shape)

    mask[x_mid - size:x_mid+size, y_mid-size:y_mid+size] = 1

    mask = torch.unsqueeze(mask, axis=0).repeat(3, 1, 1)

    noise_shape[0] = stage1_trial

    noise = torch.rand(noise_shape)
    noise = 8 * (noise - 0.5)
    #noise = T.Resize(input_size)(noise)

    ### find pattern ###
    
    for j in range(len(noise)):
        mask = mask.cpu()
        img_batch = img_batch.cpu()
        noise[j] = noise[j].cpu()
        perturbed = (1 - mask) * img_batch + mask * noise[j]

        mask1 = mask.detach().cpu().numpy().transpose(1,2,0)
        cv2.imwrite('processing_images/maskhardlabel.jpg',mask1[:,:,::-1]*255)
        
        mask1 = noise[j].detach().cpu().numpy().transpose(1,2,0)
        cv2.imwrite('processing_images/patchhardlabel.jpg',mask1[:,:,::-1]*255)
        
        mask1 = perturbed.squeeze(0).detach().cpu().numpy().transpose(1,2,0)
        cv2.imwrite('processing_images/imagehardlabel.jpg',mask1[:,:,::-1]*255)
        perturbed = torch.clamp(perturbed, 0, 1)
        perturbed = perturbed.to(device)
        detloss,applied_mask,applied_image,applied_patch = forward_step_origin(perturbed,patches,image_size,mask,first=False)
        
        
        
        if detloss < fmax:

            fmax = detloss
            init_mask = mask
            init_pattern = noise[j]
            #init_predict = preds
            idx = j
        '''else:
            init_pattern = noise[j]
            #init_predict = preds
            idx = j'''





    
    init_pattern = init_pattern[idx]

    perturbed = (1 - init_mask) * img_batch + init_mask * init_pattern
    perturbed = torch.clamp(perturbed, 0, 1)
    detloss,applied_mask,applied_image,applied_patch = forward_step_origin(perturbed,patches,image_size,mask,first=False)
            
    favg = 0
    for i in range(stage1_total - stage1_iter * stage1_trial):
        noise = torch.rand(noise_shape[1:])
        noise = 8 * (noise - 0.5)
        noise = T.Resize(input_size)(noise)

        perturbed = (1 - init_mask) * img_batch + init_mask * (init_pattern + noise)
        perturbed = torch.clamp(perturbed, 0, 1)

        detloss,applied_mask,applied_image,applied_patch = forward_step_origin(perturbed,patches,image_size,mask,first=False)

        
        

        if detloss > fmax:
            fmax = detloss
            init_pattern = init_pattern + noise

    perturbed = (1 - init_mask) * img_batch + init_mask * init_pattern
    perturbed = torch.clamp(perturbed, 0, 1)
    detloss,applied_mask,applied_image,applied_patch = forward_step_origin(perturbed,patches,image_size,mask,first=False)
    
    last_favg = fmax
    last_fval = []
    fmax = fmax.cpu()
    hist_favg = [fmax]
    hist_pattern = [pattern]





    mask = init_mask
    last_loss = fmax
    noise_shape_inter = img_batch.repeat(noise_number,1,1,1)
    noise_shape = noise_shape_inter.shape

    for i in range(0, n_queries):
        mask = init_mask
        total_it = i
        num = len(hist_favg)
        k = np.minimum(4, num)
        
        topk = torch.topk(torch.stack(hist_favg), k)  #asr topk个asr

        topk_value = topk[0].detach().cpu().numpy()
        topk_prob = np.exp(topk_value * 5)   # e^5 
        topk_prob /= np.sum(topk_prob)  #归一化
        
        topk_index = topk[1].detach().cpu().numpy()
        #print('topk_index=',topk_index)
        selected = np.random.choice(topk_index, 1, replace=False, p=topk_prob)[0]  #根据概率选一个index

        prob = (hist_favg[selected] + 1) / ((1 - last_loss) + 1 + eps)  #该index的asr/上一次的asr
        
        if torch.rand(1)[0] <= prob:        #0~1的随机数，小于prob则选择  #修改后 根据当前的来看
            pattern = hist_pattern[selected]
        '''if count == 15 :
            pattern = torch.rand_like(pattern)
            best_loss = loss_init
            count = 0'''
        
        noise = torch.rand(noise_shape)     # （30,3,h,w）
        noise = 2.0 * (noise - 0.5)   #(-1,1)
        noise = T.Resize(input_size)(noise)
        
        ### normalize perturbation ###
        if favg_diff < 0:  #如果是负增长(-)
            favg_diff = favg_diff.cpu()
            scale = np.maximum(0.9 + favg_diff // 0.2, 0.8) #0.8~0.9
            noise *= torch.exp(noise.abs() - 1) * scale     # noise: (e^n -1) * 0.8

        ### update gradient sign ###
        if signs is not None:
            if favg_diff > 0:  #正向
                n = int(favg_diff / 0.1) - 1 #0~10 -1 : -1~9 大概率是正数
            else:   #负向增长   
                n = - int(favg_diff / 0.1)  #   负数变正数
                signs = - signs     #负方向

            n = min(noise_number - i, n) 
            if n > 0:
                idx = np.random.choice(np.arange(i, noise_number), n, replace=False)  #从i到30取n个数
                noise[idx] = noise[idx].abs() * signs #
        
        noise = noise.to(device)
        trigger = pattern + noise

        if i > 3:
            for t in range(np.minimum(i, noise_number)):  #（0,i）
                choice1 = np.random.choice(topk_index, 1)[0]
                prob = nn_graph[choice1][:num]      #nn_graph 是 cosine矩阵，先根据topk概率选择第一个，然后根据第一个的cos相似度矩阵选择第二个pattern
   
                prob[np.isnan(prob)] = 0
                prob = prob * (np.array(hist_favg) + 1)
                if np.sum(prob) == 0:
                    prob[...] = 1
                prob = prob / np.sum(prob)
                
                choice2 = np.random.choice(num, 1, p=prob)[0]

                eta = torch.rand(1)[0]
                new_pattern = eta * hist_pattern[choice1]\
                                + (1 - eta) * hist_pattern[choice2]
                trigger[t] = new_pattern + noise[t] * 0.05
        trigger = trigger.to(device)
        mask = mask.to(device)
        img_batch = img_batch.to(device)
        trigger = mask * trigger
        

        favg = 0
        grads = []
        
        perturbed = (1 - mask) * img_batch + trigger
        
        perturbed = torch.clamp(perturbed, 0, 1)
        rv = perturbed - img_batch - mask * pattern
        
        detec_loss_sum = 0
        detec_loss_list = []
        #decisions = decision_function(model, perturbed, normalize, target)
        for j in range(len(perturbed)):
            sub_trigger = trigger[j].unsqueeze(0)
            detloss,applied_mask,applied_image,applied_patch = forward_step_origin(img_batch,sub_trigger,image_size,mask,first=False)
            detec_loss_sum+=detloss
            detec_loss_list.append(detloss.cpu())

            ### check asr improvement compared to the last time ###
        loss  = detec_loss_sum/len(perturbed)  #求30个噪声造成的平均值

        diff = (loss - last_loss) #当此平均值-上一次平均值，差值越多说明方向越对
        if torch.abs(diff) < 0.01 and torch.abs(diff) > 0:
            diff = diff * 100
        elif torch.abs(diff) <0.1 and torch.abs(diff) >0.01:
            diff = diff * 10
        
        ### rescale gradient magnitude for different samples ###
        if diff <= 0:
            diff = -diff
            diff = torch.exp(diff)  #放大这个方向
            weight = diff.abs()
        
        elif diff > 0:
            diff1 = torch.log(1+diff)
            weight = diff1    #缩小这个方向
        #print('loss=',loss)   
        loss = loss.to(device)
        
        detec_loss_list = torch.from_numpy(np.array(detec_loss_list)).to(device)

        loss_diff = -(detec_loss_list - loss)
        loss_diff = loss_diff.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,rv.shape[1] ,rv.shape[2],rv.shape[3])
        gradf = torch.mean(loss_diff * rv, dim=0)

        gradf = gradf * weight / torch.linalg.norm(gradf)
        gradf = gradf.unsqueeze(0)
        favg_diff = -(loss - last_loss)

        ### record gradient sign for the next iteration ###
        if (loss < last_loss ):
            signs = torch.sign(gradf)
        else:
            signs = None
        if len(mask.shape)==3:
            mask = mask.unsqueeze(0)

        gradf = gradf[mask > 0]
        
        
        gradf_flat = gradf.flatten()

        ### adam optimizer ###
        if i == 0:
            grad_momentum = gradf
            full_matrix   = torch.outer(gradf_flat, gradf_flat)
        else:
            grad_momentum = beta1 * grad_momentum + (1 - beta1) * gradf
            full_matrix   = beta2 * full_matrix\
                            + (1 - beta2) * torch.outer(gradf_flat, gradf_flat)

        grad_momentum /= (1 - beta1 ** (i + 1))
        full_matrix   /= (1 - beta2 ** (i + 1))
        factor = 1 / torch.sqrt(eps + torch.diagonal(full_matrix))
        gradf = (factor * grad_momentum.flatten()).reshape_as(gradf)

        if loss > last_loss:
            gradf *= torch.min(0.1 / (loss - last_loss), torch.tensor(0.5))

        
        
        gradf = gradf.unsqueeze(0)
        
        mask = mask.float()
        pattern = pattern.float()
        pattern[mask > 0] = pattern[mask > 0].float() + (gradf * lr).float()
        
        
        ### record neighboring triggers ###
        for t in range(num):
            p1 = hist_pattern[t].flatten()
            p2 = (pattern * mask).flatten()
            sim = (F.cosine_similarity(p1, p2, dim=0) + 1) / 2
           
            nn_graph[t, i+1] = sim
            nn_graph[i+1, t] = sim
        score = (1-loss).cpu()
        if loss < 0 or loss > 5:
            return perturbed,loss,total_it
        if torch.isnan(loss).any():
            return perturbed,loss,total_it
        score = score.cpu()
        hist_favg.append(score)
        hist_pattern.append(pattern)

        last_loss = loss
        
        print('iter=',i,'image_size=',image_size[0],'loss=',loss,'best_loss=',best_loss,'count=',count,'lr=',lr)
        perturbed = (1 - mask) * img_batch + mask * pattern
        perturbed = torch.clamp(perturbed, 0, 1)
        loss_list.append(loss)

        if loss < best_loss:
            best_loss = loss
            best_img = perturbed
        
        last_perturbed = perturbed
        if loss < threshold:
            return perturbed,loss,total_it,best_img
    
    return last_perturbed,best_loss,total_it,best_img








def gradient_color(forward_step,img_batch,patches,n_queries,it,image_size,applied_mask,loss,device,threshold,do_aug):
    loss_list = []
    loss_ind = []
    lr = 0.3
    eps = 1e-10
    query_color = 5000
    grad_momentum = 0
    full_matrix = 0
    beta1 = 0.9
    beta2 = 0.999
    m = 0
    v = 0
    color_num = 2
    img_batch = F.interpolate(img_batch,(image_size[0],image_size[1]))
    patches = F.interpolate(patches,(image_size[0],image_size[1]))
    applied_mask = F.interpolate(applied_mask,(image_size[0],image_size[1]))
    loss_init,applied_mask,applied_image,applied_patch = forward_step(img_batch,patches,image_size,first=False)
    print('loss_init=',loss_init,'n_queries=',n_queries)
    if loss_init < threshold:
        return patches,loss_init
    mask_find = applied_mask[0,0]
    coor = torch.nonzero(mask_find)
    y_coor = coor[:,0]
    x_coor = coor[:,1]
    patches = patches[:,:3,:,:]
    
    x_min = x_coor.min().item()
    x_max = x_coor.max().item()
    y_min = y_coor.min().item()
    y_max = y_coor.max().item()
    loss_min = loss_init
    
    h_u = int(np.ceil((y_max - y_min)/(color_num-1)))
    best_patch = None
    
    #initialize color
    
    best_patch, loss_min, best_color,itera = color_selected(forward_step,img_batch,color_num,image_size,patches,y_min,y_max,threshold,n_queries*0.9,device)
    
    if loss_min < threshold:
        return patches, loss_min
    
    
    '''for i in range(color_num-1):
        for h in range(h_u+1):
            for channel in range(3):
                height = h_u * i + h + y_min
                patches[:,channel,height,:] = best_color[i][channel] + (best_color[i+1][channel]-best_color[i][channel]) * h/h_u'''
    
    patches_new = best_patch
    detloss_pos,applied_mask,applied_image,applied_patch = forward_step(img_batch,best_patch,image_size,first=False)
    patches = patches_new.clone()
    print('loss=',detloss_pos)
    num_query = 0
    t = 0
    count = 0
    lamda = 0.4
    ###### optimize ##########
    while num_query <= (n_queries-n_queries*0.9):
        patches_pos = patches.clone()
        patches_neg = patches.clone()
        patches_gradient = torch.zeros_like(patches)
        
        lamda = random.uniform(0.1, 0.5)
        for height in range(y_min,y_max+1):
            for channel in range(3):
                if height != y_max:
                    optimization_direction = patches[:,channel,height+1,int((x_max-x_min)/2)] - patches[:,channel,height,int((x_max-x_min)/2)]
                else:
                    optimization_direction = (patches[:,channel,height,int((x_max-x_min)/2)] - patches[:,channel,height-1,int((x_max-x_min)/2)])
                if optimization_direction == 0:
                    continue
                patches_pos[:,channel,height,:] = patches[:,channel,height,:] + lamda*optimization_direction
                patches_neg[:,channel,height,:] = patches[:,channel,height,:] - lamda*optimization_direction
                
                
            detloss_pos,applied_mask,applied_image,applied_patch = forward_step(img_batch,patches_pos,image_size,first=False)
            detloss,applied_mask,applied_image,applied_patch = forward_step(img_batch,patches,image_size,first=False)
            detloss_neg,applied_mask,applied_image,applied_patch = forward_step(img_batch,patches_neg,image_size,first=False)
            num_query = num_query + 3
            if detloss_pos < detloss and detloss_pos < detloss_neg:
                patches_new[:,:,height,x_min:x_max] = patches[:,:,height,x_min:x_max] + lamda*optimization_direction
            elif detloss_neg < detloss_pos and detloss_neg < detloss:
                patches_new[:,:,height,x_min:x_max] = patches[:,:,height,x_min:x_max] - lamda*optimization_direction
            elif detloss <= detloss_neg and detloss <= detloss_pos:
                patches_new[:,:,height,x_min:x_max] = patches[:,:,height,x_min:x_max]

            
            '''for channel in range(3):
                optimization_direction = patches[:,channel,height+1,int((x_max-x_min)/2)] - patches[:,channel,height-1,int((x_max-x_min)/2)]
                if optimization_direction == 0:
                    patches_gradient[:,channel,height,:] = 0
                else:
                    grad = (detloss_pos - detloss_neg) / (0.2*optimization_direction)
                    patches_gradient[:,channel,height,:] = grad'''
                    
        
        
        
        color_unit = [0,0,0]
        for i in range(color_num-1):
            for h in range(h_u):
                for channel in range(3):
                    color_unit[channel] = (best_color[i+1][channel] - best_color[i][channel])/h_u
                    height = h_u*i + h + y_min
                    patches_new[:,channel,height,:] = torch.clamp(patches_new[:,channel,height,:], patches[0,channel,height,0].cpu().numpy()-color_unit[channel], patches[0,channel,height,0].cpu().numpy()+color_unit[channel])


        

        
        patches_new = torch.clamp(patches_new, 0, 1)
        
        mask = applied_mask.squeeze(0).permute(1,2,0).cpu().numpy()
        save_patch = F.interpolate(patches_new,(applied_image.shape[-2],applied_image.shape[-1]))
        save_patch = save_patch.squeeze(0).cpu().detach().numpy().transpose(1,2,0)
        #save_patch_mask = np.expand_dims(save_patch[:,:,3],axis=2)
        save_patch = save_patch[:,:,:3]
        cv2.imwrite('save_patches/save_patch_gradient_color.jpg',save_patch[:,:,::-1]*mask*255)
        loss,applied_mask,applied_image,applied_patch = forward_step(img_batch,patches_new,image_size = image_size,first=False)
        
        patches = patches_new
        
        
        if loss<= loss_min:
            loss_min = loss
            best_patch = patches_new
            mask = applied_mask.squeeze(0).permute(1,2,0).cpu().numpy()
            save_patch = F.interpolate(best_patch,(applied_image.shape[-2],applied_image.shape[-1]))
            save_patch = save_patch.squeeze(0).cpu().detach().numpy().transpose(1,2,0)
            #save_patch_mask = np.expand_dims(save_patch[:,:,3],axis=2)
            save_patch = save_patch[:,:,:3]
            cv2.imwrite('save_patches/save_patch_gradient.jpg',save_patch[:,:,::-1]*mask*255)
            
        #loss_min[idx_to_update] = loss[idx_to_update]
        total_loss = loss.item()

        
        print('the iteration:%d,image size:%d, min loss:%f,total loss:%f,lr:%f'
            %(num_query,image_size[0],loss_min,total_loss,lr))
        
        if loss_min < threshold:
            break
        
    
    
    
        
    print('final result: image size:%d,min loss:%f,lr:%f'%(image_size[0],loss_min,lr))
    return patches, loss_min




def color_selected2(forward_step,img_batch,num_color,image_size,patches,adv_patch_hsv,x_min,x_max,y_min,y_max,threshold,num_query,device):
    adv_patch_hsv
    color_candidata_num = 10
    step = int(num_query/(color_candidata_num * (num_color-1)))
    print('step=',step)
    D=3    #单条染色体基因数目
    G = 100
    Rm = 0.5
    Rc = 0.6
    best_image = img_batch
    number_best = 2
    loss_min = 2
    best_patch = patches
    #color initialization
    #select the color of hue value in hsv space,   hsv --> rgb --> hsv --> rgb
    color_candidata = []
    for num in range(color_candidata_num):
        color = []
        for num2 in range(num_color):
            color.append([random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1)])
        color_candidata.append(color)
    color_candidata = np.array(color_candidata)
    for i in range(step):
        selected_block = []
        NP = len(color_candidata) # 3 个
        #print('all_blocks_shape=',np.array(all_blocks).shape)
        
        #变异操作，和遗传算法的变异不同！,得到任意两个个体的差值，与变异算子相乘加第三个个体
        
        for m in range(color_candidata_num):
            r1=m
            r2=random.sample(range(0,NP),1)[0]
            r3=random.sample(range(0,NP),1)[0]
            while r2 == r1:
                r2=random.sample(range(0,NP),1)[0]
            
            while r3 == r2 or r3 == r1:
                r3=random.sample(range(0,NP),1)[0]
            

            for num3 in range(num_color):
                Parents_sub = color_candidata[r1][num3].copy()
                color_candidata[r1][num3] =  color_candidata[r1][num3]+Rm*(color_candidata[r2][num3]-color_candidata[r3][num3])  # num  
                Children_sub = color_candidata[r1][num3]
                
                Children_sub_clip = np.clip(Children_sub,0,1)
                for num4 in range(3):
                    rand = random.random()
                    if rand <= Rc:
                        color_candidata[r1][num3][num4] = Parents_sub[num4]
                    else:
                        color_candidata[r1][num3][num4] = Children_sub_clip[num4]
                
                '''rand = random.random()
                if rand <= Rc:
                    color_candidata[r1][num3] = Parents_sub
                else:
                    color_candidata[r1][num3] = Children_sub_clip'''
            #best_colors[r1][1] = best_colors[r1][1]+Rm*(color_candidata[r2][1]-color_candidata[r3][1])  #color
            #best_colors[r1][2] = best_colors[r1][2]+Rm*(color_candidata[r2][2]-color_candidata[r3][2])
            
            Children = color_candidata[r1]
            
            
            color_num = num_color
            rgb = [0,0,0]
            h_u = int(np.ceil((y_max - y_min)/(color_num-1)))
            for i2 in range(color_num-1):
                for h in range(h_u+1):
                    for channel in range(3):
                        height = h_u * i2 + h + y_min
                        color1 = torch.tensor(Children[i2][channel]).to(device)
                        color2 = torch.tensor(Children[i2+1][channel]).to(device)
                        
                        #patches[:,channel,height,:] = color1 + (color2-color1) * h/h_u
                        ##convert to real mask
                        rgb[channel] = color1 + (color2-color1) * h/h_u
                        r = rgb[0]
                        g = rgb[1]
                        b = rgb[2]
                        hsv_color = rgb_to_hsv(r,g,b)
                        hue = hsv_color[0]
                        adv_patch_hsv[:,0,int(height),:] = hue
                        adv_patch_hsv[:,1,int(height),:] = hsv_color[1]
                        #adv_patch_hsv[:,2,int(height),:] = hsv_color[2]
                        
            patch_mask = adv_patch_hsv[:,3,:,:].unsqueeze(0)
            patch_rgb_img = cv2.cvtColor(adv_patch_hsv[:,:3,:,:].squeeze(0).cpu().numpy().transpose(1,2,0),cv2.COLOR_HSV2RGB)
            cv2.imwrite('processing_images/mask_processing.jpg',patch_rgb_img*255)
            patches = torch.from_numpy(patch_rgb_img).unsqueeze(0).permute(0,3,1,2).to(device)
            patches = torch.cat((patches,patch_mask),1)
            
            patches_new = patches.clone() 
            loss,applied_image,applied_patch= forward_step(img_batch,patches_new,image_size = image_size,first=False)
            total_loss = loss# + diff_color*0.1 + v_value1*0.01 + v_value2*0.01
            if total_loss < loss_min:
                loss_min = total_loss
                loss_min_det = loss
                best_patch = patches_new
                best_color = Children
                best_image = applied_image
            print('num_color_current=',num_color,'step=',i*color_candidata_num,'detloss_min=',loss_min_det,'loss_min',loss_min,'det_loss=',loss,'total_loss=',total_loss)
            itera = i*color_candidata_num
            if loss_min <threshold:
                return best_patch, loss_min, best_color, best_image
    
    
    return best_patch, loss_min, best_color,best_image


def color_selected(forward_step,forward_step_thermal,img_batch,num_color,image_size,patches,applied_image,thermal_image,rgb_mask,adv_patch_hsv,applied_image_hsv,x_min,x_max,y_min,y_max,threshold,num_query,device,harmonic,model):
    adv_patch_hsv
    color_candidata_num = 10

    step = int(num_query/(color_candidata_num * (num_color-1)))

    D=3   
    G = 100
    Rm = 0.5
    Rc = 0.6
    best_image = img_batch
    number_best = 2
    loss_min = 2
    best_patch = patches

    
    patch_mask = adv_patch_hsv[:,3,:,:].unsqueeze(0)
    adv_patch_hsv_new = adv_patch_hsv.clone()
    applied_image_hsv_new = applied_image_hsv.clone().to(device)
    patch_mask[patch_mask<0.5] = 0
    patch_mask[patch_mask>=0.5] = 1
    if harmonic == 'direct':
        select_angle = 93.2   # 9 or 93.2 in oppositive color
    elif harmonic == 'analogous':
        select_angle = 90

    color_candidata = []
    
    
    
    loss_thermal,img_attack2 = forward_step_thermal(thermal_image)
    # DE algorithm
    for num in range(color_candidata_num):
        color = []
        color_rgb1 = [random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1)]#,random.uniform(0-v_min,1-v_max)]
        color.append(color_rgb1)
        for num2 in range(num_color):
            color_hsv = rgb_to_hsv(color_rgb1[0],color_rgb1[1],color_rgb1[2])
            hue1,s1,v1 = color_hsv[0], color_hsv[1],color_hsv[2]
            if harmonic == 'direct':
                hue_2 = random.uniform(hue1-select_angle,hue1+select_angle)
                hue_3 = random.uniform((hue1-select_angle+180)%360,(hue1+select_angle+180)%360)
                rand_1 = random.random()
                if rand_1 < 0.5:
                    hue2 = hue_2
                else:
                    hue2 = hue_3
            elif harmonic == 'analogous':
                hue2 = random.uniform(hue1-select_angle,hue1+select_angle)
            s2 = s1
            v2 = v1

            color_rgb2 = hsv_to_rgb(hue2,s2,v2)
            #color_rgb2.append(random.uniform(0-v_min,1-v_max))
            color.append(color_rgb2)

        color_candidata.append(color)
    
    color_candidata = np.array(color_candidata)
    
   
    for i in range(step):
        selected_block = []
        NP = len(color_candidata) # 3 个
        #print('all_blocks_shape=',np.array(all_blocks).shape)
        
        for m in range(color_candidata_num):
            r1=m
            r2=random.sample(range(0,NP),1)[0]
            r3=random.sample(range(0,NP),1)[0]
            while r2 == r1:
                r2=random.sample(range(0,NP),1)[0]
            
            while r3 == r2 or r3 == r1:
                r3=random.sample(range(0,NP),1)[0]
            

            for num3 in range(num_color):
                
                Parents_color = color_candidata[r1][num3][:3].copy()
                color_candidata[r1][num3][:3] =  color_candidata[r1][num3][:3]+Rm*(color_candidata[r2][num3][:3]-color_candidata[r3][num3][:3])  # num  
                Children_color = color_candidata[r1][num3][:3]
                
                Children_color_clip = np.clip(Children_color,0,1)
      
                
   
                rand = random.random()
                if rand <= Rc:
                    color_candidata[r1][num3][:3] = Parents_color
                    #color_candidata[r1][num3][3] = Parents_v
                else:
                    color_candidata[r1][num3][:3] = Children_color_clip
                    #color_candidata[r1][num3][3] = Children_v_clip
            
                if num3 == 0 :
                    hue1, s1,v1 = rgb_to_hsv(color_candidata[r1][0][0],color_candidata[r1][0][1],color_candidata[r1][0][2])
                
                hue2, s2,v2 = rgb_to_hsv(color_candidata[r1][num3][0],color_candidata[r1][num3][1],color_candidata[r1][num3][2])
                if harmonic == 'direct':
                    if (hue2 <= (hue1 + select_angle)%360 and hue1 >= (hue1 - select_angle)%360) or (hue2 <= (hue1 + select_angle +180)%360 and hue1 >= (hue1 - select_angle + 180)%360):
                        hamonic_color = 1
                    else:
                        hamonic_color = 0
                elif harmonic == 'analogous':
                    if (hue2 <= (hue1 + select_angle)%360 and hue1 >= (hue1 - select_angle)%360):
                        hamonic_color = 1
                    else:
                        hamonic_color = 0
                if hamonic_color == 0:
                    hue_2 = random.uniform(hue1-select_angle,hue1+select_angle)
                    hue_3 = random.uniform((hue1-select_angle+180)%360,(hue1+select_angle+180)%360)
                    rand_1 = random.random()
                    if rand_1 < 0.5:
                        hue2 = hue_2
                    else:
                        hue2 = hue_3
                        
                    
                    color_candidata[r1][num3][:3] = hsv_to_rgb(hue2,s2,v2)
                    
   
            Children = color_candidata[r1]
            
            Children = [[0, 200, 0],[200, 200, 0]]
            color_num = num_color
            rgb = [0,0,0]
            h_u = int(np.ceil((y_max - y_min)/(color_num-1)))
            
            for i2 in range(color_num-1):
                for h in range(h_u+1):
                    
                        height = h_u * i2 + h + y_min
                        color1 = torch.tensor(Children[i2]).to(device)
                        color2 = torch.tensor(Children[i2+1]).to(device)

                        rgb = color1 + (color2-color1) * h/h_u
                        r = rgb[0]
                        g = rgb[1]
                        b = rgb[2]
                        hsv_color = rgb_to_hsv(r,g,b)
                        applied_image_hsv_new[:,0,int(height),:] = hsv_color[0]
                        applied_image_hsv_new[:,1,int(height),:] = hsv_color[1]
 
            adv_patch_hsv_new[:,2:,:] = adv_patch_hsv[:,2:,:]
                        
          
            patch_rgb_img = cv2.cvtColor(applied_image_hsv_new[:,:3,:,:].squeeze(0).cpu().numpy().transpose(1,2,0),cv2.COLOR_HSV2RGB)

          
            patches = torch.from_numpy(patch_rgb_img).unsqueeze(0).permute(0,3,1,2).to(device)
            patches_new = patches.clone() 
   
            #patches = patches[:,:3,:,:]
            loss,applied_image,applied_patch,_= forward_step(img_batch,patches,image_size = image_size,first=False,get_mask=False)
            total_loss = loss# + diff_color*0.1 + v_value1*0.01 + v_value2*0.01
            if total_loss < loss_min:
                loss_min = total_loss
                loss_min_det = loss
             
                best_patch = patches_new
                best_color = Children
                best_image = applied_image
            print('model=',model,'num_color_current=',num_color,'step=',i*color_candidata_num,'detloss_min=',loss_min_det,'loss_min',loss_min,'loss_thermal=',loss_thermal,'det_loss=',loss,'total_loss=',total_loss)
            itera = i*color_candidata_num
            if loss_min <threshold:
                return best_patch, loss_min,loss_thermal, best_color, best_image
                
                
    return best_patch, loss_min,loss_thermal, best_color,best_image           
 
 
    
    





def box_determine(image,device):
    img_size = image.shape[-1]

    premodel = YoloDetector(target_size=720, device=device, min_face=90) 
    image = image.to(device)
    
    output = premodel.detector(image)[0]
    score = output[...,4]
    #score = torch.sigmoid(scores)
    max_score = torch.max(score,dim=1)
    det_loss = torch.mean(max_score.values)  
    xc = output[..., 4] > 0
    min_wh, max_wh = 2, 4096
    agnostic=False
    new_output = [torch.zeros((0, 16), device=output.device)] * output.shape[0]
    
    for num, prediction in enumerate(output):
        object_conf = torch.zeros_like(prediction[:, 15:])
        prediction = prediction[xc[num]]
        object_conf = prediction[:, 4:5]*prediction[:, 15:]
        
        box = xywh2xyxy(prediction[:, :4])
        conf, j = object_conf.max(1, keepdim=True)
        
        prediction = torch.cat((box, conf, prediction[:, 5:15], j.float()), 1)[conf.view(-1) > 0]
        c = prediction[:, 15:16] * (0 if agnostic else max_wh)  # classes
        
        boxes, scores = prediction[:, :4] + c, prediction[:, 4]  # boxes (offset by class), scores
        
        index = torchvision.ops.nms(boxes, scores, 0.4)  # NMS
        new_output[num] = prediction[index]
        
        
        '''print('prediction=',prediction)
        max_score = torch.max(prediction[index,4])
        max_score_index = torch.nonzero(prediction[index,4]==max_score)
        print('max_score=',max_score)'''
        
        sub_label = prediction[index][0]
        
        label_box = torch.zeros_like(sub_label)
        width = sub_label[2] - sub_label[0]
        height = sub_label[3] - sub_label[1]

        x_min = int(sub_label[0])
        x_max = int(sub_label[2])
        y_max = int(sub_label[3])
        y_min = int(sub_label[1])

        label_box[0] = 0
        label_box[1] = sub_label[0] + width /2
        label_box[2] = sub_label[1] + height /2
        label_box[3] = width
        label_box[4] = height
        label = label_box[...,:5]/img_size
    lab_batch = label.unsqueeze(0).unsqueeze(0).to(device)
    lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)
    lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
    lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
    lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
    lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
    
    #target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(0.25)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.25)) ** 2))
    target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(0.3)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.3)) ** 2))
    
    target_size = torch.tensor(66.0803/416*128)
    target_size = target_size.to(device)/2
    print('target_size=',target_size)
    return [x_min,y_min,x_max,y_max], int(target_size)








def stickers(forward_step,forward_step_sticker,forward_step_thermal,img_batch,thermal_image,patches,white_patch,adv_patch_hsv,n_queries,image_size,applied_mask,device,visual_threshold,thermal_threshold,do_aug,harmonic,model):
    loss_list = []
    loss_ind = []
    lr = 0.3
    eps = 1e-10
    query_color = 5000
    grad_momentum = 0
    full_matrix = 0
    beta1 = 0.9
    beta2 = 0.999
    m = 0
    v = 0
    color_num = 2
    img_batch = F.interpolate(img_batch,(image_size[0],image_size[1])).to(device)
    patches = F.interpolate(patches,(image_size[0],image_size[1]))
    applied_mask = F.interpolate(applied_mask,(image_size[0],image_size[1]))
    
    _,_,_,rgb_mask = forward_step(img_batch,white_patch,image_size,first=True,get_mask=True)
    rgb_mask = rgb_mask.to(device)
    cold_mask = torch.zeros_like(thermal_image).to(device) + 0.1
    thermal_image = thermal_image * (1-rgb_mask) + cold_mask * rgb_mask
    
    
    
    loss_init,applied_image,applied_patch,_ = forward_step(img_batch,patches,image_size,first=True,get_mask=False)
  
    
 
    
    
  
    applied_image = image_harmonic(applied_image,rgb_mask)
    

    applied_image_hsv = applied_image.clone().to(device)
    adv_patch_zeros = torch.zeros_like(applied_image_hsv)
    for height in range(applied_image.shape[-2]):
        for width in range(applied_image.shape[-1]):
            r = applied_image[0,0,height,width]
            g = applied_image[0,1,height,width]
            b = applied_image[0,2,height,width]
            hsv_color = rgb_to_hsv(r,g,b)
            h = hsv_color[0]
            s = hsv_color[1]
            v = hsv_color[2]
            applied_image_hsv[0,0,height,width] = h
            applied_image_hsv[0,1,height,width] = s
            applied_image_hsv[0,2,height,width] = v

                        
                

 

    
    mask_find = rgb_mask[0]
    coor = torch.nonzero(mask_find)
    y_coor = coor[:,0]
    x_coor = coor[:,1]
    #patches = patches[:,:3,:,:]

    x_min = x_coor.min().item()
    x_max = x_coor.max().item()
    y_min = y_coor.min().item()
    y_max = y_coor.max().item()
    
    sticker_name = 'stickers/bs12.png'
    scale = 5          
    stickerpic = Image.open(sticker_name)

    h_u = int(np.ceil((y_max - y_min)/(color_num-1)))
    scale1 = stickerpic.size[0]//20
    scale11 = stickerpic.size[0]//10
    scale2 = 5
    operate_sticker = change_sticker(stickerpic,scale1)
    opstickercv = img_to_cv(operate_sticker)
    sticker = change_sticker(stickerpic,scale2)   
    
    
    sticker_name2 = 'stickers/band-aid.png'
    stickerpic2 = Image.open(sticker_name2)

    operate_sticker2 = change_sticker(stickerpic2,scale11)
    opstickercv2 = img_to_cv(operate_sticker2)
    sticker2 = change_sticker(stickerpic2,scale2)   
    
    
    stickerpic2_array = np.array(stickerpic2)
 
    stickerpic2_array[:,:,:3] = 0.1
    stickerpic2_thermal = Image.fromarray(stickerpic2_array)
    operate_sticker2_thermal = change_sticker(stickerpic2_thermal,scale11)
    opstickercv2_thermal = img_to_cv(operate_sticker2_thermal)
    sticker2_thermal = change_sticker(stickerpic2_thermal,scale2)   
    
    zstore = generate_zstore(img_batch)

    zstore = np.expand_dims(zstore,axis=-1)
    facemask = make_mask(img_batch)
    facemask1 = np.zeros_like(facemask)

    num_space = np.sum(facemask).astype(int)
    searchspace_upper = []
    
    num_space = np.sum(np.array(mask_find.cpu())).astype(int)
    searchspace_mask = []
    
    coor = torch.nonzero(torch.from_numpy(facemask))
    y_coor = coor[:,0]
    x_coor = coor[:,1]
    #patches = patches[:,:3,:,:]

    x_min_face = x_coor.min().item()
    x_max_face = x_coor.max().item()
    y_min_face = y_coor.min().item()
    y_max_face = y_coor.max().item()
    
    

    
    face_mask_test = np.zeros_like(facemask)
    for i in range(facemask.shape[0]):
        if i > (y_max_face-y_min_face)*0.43+y_min_face:
            break
        for j in range(facemask.shape[1]):
            if(facemask[i][j] == 1):
                facemask1[i,j] = 1
                searchspace_upper.append([j,i])
                
                # pack_searchspace[i][j] = k
            

    mask_rgb = rgb_mask[0]

    coor = torch.nonzero(mask_rgb)
    y_coor = coor[:,0]
    x_coor = coor[:,1]
    #patches = patches[:,:3,:,:]

    x_min_rgb = x_coor.min().item()
    x_max_rgb = x_coor.max().item()
    y_min_rgb = y_coor.min().item()
    y_max_rgb = y_coor.max().item()
         
    for i in range(x_min_rgb,x_max_rgb):
        for j in range(y_min_rgb,y_max_rgb):
            if(mask_find[i][j] == 1):
                searchspace_mask.append([j,i])
                # pack_searchspace[i][j] = k
   

    percentage_of_color = 0.5
    
    best_patch, loss_visual_min,loss_thermal_min, best_color,applied_image = color_selected(forward_step,forward_step_thermal,img_batch,color_num,image_size,patches,applied_image,thermal_image,rgb_mask,adv_patch_hsv,applied_image_hsv,x_min,x_max,y_min,y_max,visual_threshold,n_queries*percentage_of_color,device,harmonic,model)
    
    if loss_visual_min < visual_threshold and loss_thermal_min < thermal_threshold:
        return patches,loss_visual_min,loss_thermal_min,applied_image

    #img_attack, loss_min,best_img = sticker_optimization(forward_step_sticker,applied_image,loss_min,best_patch,mask_find,y_min,y_max,x_min,x_max,searchspace_upper,searchspace_mask,threshold,sticker,opstickercv,sticker2,opstickercv2,zstore,n_queries*(1-percentage_of_color),image_size,device)
    
    img_attack, loss_visual_min,loss_thermal_min,best_img = band_aid_optimization(forward_step_sticker,forward_step_thermal,applied_image,thermal_image,mask_find,y_min,y_max,x_min,x_max,searchspace_upper,searchspace_mask,visual_threshold,thermal_threshold,sticker,opstickercv,sticker2,opstickercv2,sticker2_thermal,opstickercv2_thermal,zstore,n_queries*(1-percentage_of_color),image_size,device)
 
    
    #return patches,1,1,applied_image
    
    return patches,loss_visual_min,loss_thermal_min,applied_image



def perturb_image(backimg,sticker,opstickercv,search_space,magnification,zstore,params):
    
    backimg = backimg.cpu().squeeze(0).numpy().transpose(1,2,0)*255
    imgs = []
    if len(params) == 2:
        params = [params]

    for param in params:

        if int(param[0]) >= len(search_space):
            param[0] = len(search_space) - 1
        x = int(search_space[int(param[0])][0])
        y = int(search_space[int(param[0])][1])
        angle = param[1]
        rt_sticker = rotate_bound_white_bg(opstickercv, angle)
      
        nsticker,_ = deformation3d(sticker,rt_sticker,magnification,zstore,x,y)
       
        backimg = make_stick2(backimg=backimg, sticker=nsticker, x=x, y=y)#, factor=xs[i][1])
        save_image = np.array(backimg)
        
    return np.array(backimg)/255





def rotate_bound_white_bg(imagecv, angle):

    (h, w) = imagecv.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    

    rotated = cv2.warpAffine(imagecv, M, (w, h),borderValue=(255,255,255,0))
    b, g, r, a = cv2.split(rotated)
    rotated_array = cv2.merge([r, g, b, a])
    rt_sticker = Image.fromarray(np.uint8(rotated_array))
    return rt_sticker




def sticker_optimization(forward_step_sticker,img_batch,loss_min,patches,mask_find,y_min,y_max,x_min,x_max,searchspace_upper,searchspace_mask,threshold,sticker,opstickercv,sticker2,opstickercv2,zstore,queries,image_size,device):
    
    candidate_num = 10
    sticker_num = 3
    step = int(queries/(candidate_num))
    print('step=',step)
    D=3    #
    G = 100
    Rm = 0.5
    Rc = 0.6
    best_img = img_batch
    number_best = 2
    loss_min = loss_min
    best_patch = patches
    best_attack = img_batch
    #color initialization
    #y_min = y_min + 0.25*y_min
    y_max = y_max - 0.25*y_max
    x_min = x_min + 0.25*x_min
    x_max = x_max - 0.25*x_max
    candidate = []
    for num in range(candidate_num):
        sub_candidate = []
        for num2 in range(sticker_num):
            #sub_candidate.append([random.uniform(x_min, x_max),random.uniform(y_min, y_max),random.uniform(0, 360)])
            sub_candidate.append([int(random.uniform(0, len(searchspace_mask)-1)),random.uniform(0, 360)])
        candidate.append(sub_candidate)
    candidate = np.array(candidate)

    for i in range(step):
        selected_block = []
        NP = len(candidate) # 3 个
        #print('all_blocks_shape=',np.array(all_blocks).shape)

        for m in range(candidate_num):
            r1=m
            r2=random.sample(range(0,NP),1)[0]
            r3=random.sample(range(0,NP),1)[0]
            while r2 == r1:
                r2=random.sample(range(0,NP),1)[0]

            while r3 == r2 or r3 == r1:
                r3=random.sample(range(0,NP),1)[0]
            
           
            for num3 in range(sticker_num):
                Parents_sub = candidate[r1][num3].copy()
                candidate[r1][num3] =  candidate[r1][num3]+Rm*(candidate[r2][num3]-candidate[r3][num3])  # num  
                Children_sub = candidate[r1][num3]
              
                for num4 in range(2):
                    if num4 == 0:
                        Children_sub_clip = np.clip(Children_sub[num4],0, len(searchspace_mask)-1)
        
                    '''if num4 == 1:
                        Children_sub_clip = np.clip(Children_sub[num4],y_min,y_max)'''
                    if num4 == 1:
                        Children_sub_clip = np.clip(Children_sub[num4],0,360)
                    rand = random.random()
                    if rand <= Rc:
                        candidate[r1][num3][num4] = Parents_sub[num4]
                        
                    else:
                        candidate[r1][num3][num4] = Children_sub_clip
                
                        
                        
                
                        
                      
                '''while mask_find[int(Children_sub[0]),int(Children_sub[1])] == 0:
                    Children_sub[0] = random.uniform(y_min, y_max)
                    Children_sub[1] = random.uniform(x_min, x_max)'''
                '''rand = random.random()
                if rand <= Rc:
                    color_candidata[r1][num3] = Parents_sub
                else:
                    color_candidata[r1][num3] = Children_sub_clip'''
            #best_colors[r1][1] = best_colors[r1][1]+Rm*(color_candidata[r2][1]-color_candidata[r3][1])  #color
            #best_colors[r1][2] = best_colors[r1][2]+Rm*(color_candidata[r2][2]-color_candidata[r3][2])

            Children = candidate[r1]
            
            #Childre_new = aug_sticker(Children)
            '''img1 = perturb_image(best_patch,sticker,Children)
            img1 = torch.from_numpy(img1).permute(2,0,1).unsqueeze(0).to(device).float()
            
            loss1,img_attack1 = forward_step_sticker(img_batch,img1,image_size)
            '''
            #scale2 = 12
            magnification = 1
            
            img2 = perturb_image(img_batch,sticker,opstickercv,searchspace_mask,magnification,zstore,Children)
            img2 = torch.from_numpy(img2).permute(2,0,1).unsqueeze(0).to(device).float()
 

            
            
            loss2,img_attack2 = forward_step_sticker(img_batch,img2,mode='paste')
            '''applied_image11 = img2.squeeze(0).cpu().numpy().transpose(1,2,0)
            cv2.imwrite('processing_images/best_img_color_reason_yqc11.jpg',applied_image11[:,:,::-1]*255)'''
            
            
            best_attack = img_attack2
            '''if loss1 >= loss2:
                loss = loss2
                img = img2
                best_children = Childre_new
                attack = img_attack1
            else:
                loss = loss1
                img = img1
                best_children = Children
                attack = img_attack1'''
            if loss2 < loss_min:
                loss_min = loss2
                best_img = img2
                best_color = Children
                best_attack = img_attack2
            print('step=',i*candidate_num,'loss=',loss2,'loss_min',loss_min)
            
            if loss_min <threshold:
                return best_attack, loss_min,best_img
    
    
    return best_attack, loss_min,best_img




def band_aid_optimization(forward_step_sticker,forward_step_thermal,img_batch,thermal_image,mask_find,y_min,y_max,x_min,x_max,searchspace_upper,searchspace_mask,visual_threshold,thermal_treshold,sticker,opstickercv,sticker2,opstickercv2,sticker_thermal,opstickercv2_thermal,zstore,queries,image_size,device):
    best_img_visual = img_batch
    candidate_num = 10
    step = int(queries/(candidate_num))
    print('queries=',queries)
    D=3    #
    G = 100
    Rm = 0.5
    Rc = 0.6
    best_img = img_batch
    number_best = 2
    loss_min = 2
    
    best_attack = img_batch
    loss_visual_min = 1
    loss_thermal_min = 1
    candidate = []
    for num in range(candidate_num):
        candidate.append([int(random.uniform(0, len(searchspace_upper)-1)),random.uniform(0, 360)])
    candidate = np.array(candidate)

    for i in range(step):
        selected_block = []
        NP = len(candidate) # 3 个
        #print('all_blocks_shape=',np.array(all_blocks).shape)

        for m in range(candidate_num):
            r1=m
            r2=random.sample(range(0,NP),1)[0]
            r3=random.sample(range(0,NP),1)[0]
            while r2 == r1:
                r2=random.sample(range(0,NP),1)[0]

            while r3 == r2 or r3 == r1:
                r3=random.sample(range(0,NP),1)[0]
                Parents_sub = candidate[r1].copy()
                candidate[r1] = candidate[r1]+Rm*(candidate[r2]-candidate[r3])  # num  
                Children_sub = candidate[r1]
               
                for num4 in range(2):
                    if num4 == 0:
                        Children_sub_clip = np.clip(Children_sub[num4],0, len(searchspace_upper)-1)
        
                    if num4 == 1:
                        Children_sub_clip = np.clip(Children_sub[num4],0,360)
                    rand = random.random()
                    if rand <= Rc:
                        candidate[r1][num4] = Parents_sub[num4]
                        
                    else:
                        candidate[r1][num4] = Children_sub_clip

            Children = candidate[r1]
            
           
            magnification = 1
            
            Children = aug_sticker(Children)
 
            save_img_sub = img_batch.squeeze(0).cpu().detach().numpy().transpose(1,2,0)
            save_img_sub = cv2.resize(save_img_sub,(865,1057))
           
            save_img_sub = thermal_image.squeeze(0).cpu().detach().numpy().transpose(1,2,0)
            save_img_sub = cv2.resize(save_img_sub,(865,1057))
            
            
            img_batch_visual = perturb_image(img_batch,sticker2,opstickercv2,searchspace_upper,magnification,zstore,Children)
            img_batch_visual1 = cv2.resize(img_batch_visual,(613,741))
            
            img_batch_visual = torch.from_numpy(img_batch_visual).permute(2,0,1).unsqueeze(0).float().to(device)
            loss_visual,img_attack2 = forward_step_sticker(img_batch_visual,None)

            
            img_batch_thermal = perturb_image(thermal_image,sticker_thermal,opstickercv2_thermal,searchspace_upper,magnification,zstore,Children)
            img_batch_thermal1 = cv2.resize(img_batch_thermal,(613,741))
            
            img_batch_thermal = torch.from_numpy(img_batch_thermal).permute(2,0,1).unsqueeze(0).float().to(device)
            
            loss_thermal,img_attack2 = forward_step_thermal(img_batch_thermal)
            if loss_visual < visual_threshold:
                loss_visual = visual_threshold-0.01
            if loss_thermal < thermal_treshold:
                loss_thermal = thermal_treshold-0.01
            loss =  loss_visual + loss_thermal   #total
            #loss = loss_visual    #only consider the visual
            best_attack = img_attack2

                
            
            if loss < loss_min:
                loss_min = loss
                loss_visual_min = loss_visual
                loss_thermal_min = loss_thermal
                best_img_visual = img_batch_visual
                best_img_thermal = img_batch_thermal
                '''best_img_visual1 = best_img_visual.squeeze(0).cpu().detach().numpy().transpose(1,2,0)
                cv2.imwrite('processing_images/visual_imgbestyqc.jpg',best_img_visual1[:,:,::-1]*255)'''
                
                
                
            print('step=',i*candidate_num,'loss_min',loss_min,'loss_visual_min=',loss_visual_min,'loss_thermal_min=',loss_thermal_min,'loss=',loss,'loss_visual=',loss_visual,'loss_thermal=',loss_thermal)
            
            if loss_thermal_min< thermal_treshold and loss_visual_min< visual_threshold:
                return best_attack, loss_visual_min,loss_thermal_min,best_img_visual
    return best_attack, loss_visual_min,loss_thermal_min,best_img_visual
    

def calculate_hue_difference(color1, color2):
    color1 = color1*255
    color2 = color2*255
    
    r1 = color1[0]
    g1 = color1[1]
    b1 = color1[2]
    r2 = color2[0]
    g2 = color2[1]
    b2 = color2[2]
    hsv_color1 = rgb_to_hsv(r1,g1,b1)
    hsv_color2 = rgb_to_hsv(r2,g2,b2)
    

    hue1 = hsv_color1[0]
    hue2 = hsv_color2[0]
    

    hue_difference = abs(hue1 - hue2)
    
    return hue_difference

def vibrant_value(color):
    # 将RGB颜色转换为HSV颜色空间
    r = color[0]
    g = color[1]
    b = color[2]
    hsv_color = rgb_to_hsv(r,g,b)

    # 提取饱和度值
    saturation = hsv_color[1]

    return saturation


def rgb_to_hsv(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v


def hsv_to_rgb(h, s, v):
    if h < 0 :
        h = h +360
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = r, g, b
    return [r, g, b]



def face_landmarks(initial_pic):
    dotsets = np.zeros((1,81,2))
    detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    predictor = dlib.shape_predictor('models/shape_predictor_81_face_landmarks.dat')
    
    pic_array = np.array(initial_pic.cpu().squeeze(0).permute(1,2,0)*255)

    r,g,b = cv2.split(pic_array)
    img = cv2.merge([b, g, r])
    #img = cv2.imread(pic_dir)                          
    img = img.astype('uint8')
    imgsize = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   
    
    rects = detector(img_gray, 1)                      
    #print('num of rects=',len(rects),rects[1])
    #print(len(rects))
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[0]).parts()])
    #print(landmarks)
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])           
        #print(idx,pos)
        if(idx >= 0 and idx <= 67):
            dotsets[0][idx] = pos
        elif(idx == 78):
            dotsets[0][68] = pos
        elif(idx == 74):
            dotsets[0][69] = pos
        elif(idx == 79):
            dotsets[0][70] = pos
        elif(idx == 73):
            dotsets[0][71] = pos
        elif(idx == 72):
            dotsets[0][72] = pos
        elif(idx == 80):
            dotsets[0][73] = pos
        elif(idx == 71):
            dotsets[0][74] = pos
        elif(idx == 70):
            dotsets[0][75] = pos
        elif(idx == 69):
            dotsets[0][76] = pos
        elif(idx == 68):
            dotsets[0][77] = pos
        elif(idx == 76):
            dotsets[0][78] = pos
        elif(idx == 75):
            dotsets[0][79] = pos
        elif(idx == 77):
            dotsets[0][80] = pos

    return dotsets,imgsize

def circle_mark(facemask,dot,brw):
    dot = dot.astype(np.int16)
    dotlen = len(dot)
    for i in range(dotlen):
        x1,y1 = dot[i]
        facemask[x1,y1] = brw
        if(i == dotlen-1):
            j = 0
        else:
            j = i+1
        x2,y2 = dot[j]
        if(y2 - y1 != 0):
            k = (x2 - x1) / (y2 - y1)
            symbol = 1 if(y2 - y1 > 0) else -1
            for t in range(symbol*(y2 - y1)-1):
                y3 = y1 + symbol * (t + 1)
                x3 = int(round(k * (y3 - y1) + x1))
                # print('x1,y1,x2,y2',x1,y1,x2,y2)
                # print('x3,y3 = ',x3,y3)
                facemask[x3,y3] = brw

    dot = np.array(dot)
    lower = np.min(dot,axis = 0)[1]
    upper = np.max(dot,axis = 0)[1]
    for h in range(lower,upper+1):
        left = 0
        right = 0
        cruitl = np.min(dot,axis = 0)[0]
        cruitr = np.max(dot,axis = 0)[0]
        for i in range(cruitl-1,cruitr+2):
            if(facemask[i][h] == brw):
                left = i
                break
        for j in reversed(list(range(cruitl-1,cruitr+2))):
            if(facemask[j][h] == brw):
                right = j
                break
        left_cursor = left
        right_cursor = right
        # print('h = ',h)
        # print('left_cursor,right_cursor = ',left_cursor,right_cursor)
        if(left_cursor != right_cursor):        
            while True:
                facemask[left_cursor][h] = brw
                left_cursor = left_cursor + 1
                if(facemask[left_cursor][h] == brw):
                    break
            while True:
                facemask[right_cursor][h] = brw
                right_cursor = right_cursor - 1
                if(facemask[right_cursor][h] == brw):
                    break
    return facemask

def make_mask(initial_pic):
    dotsets,imgsize = face_landmarks(initial_pic)
    facemask = np.zeros((imgsize[1],imgsize[0]))
    #----------face--------------
    face = dotsets[0][:17]
    face2 = dotsets[0][68:]
    face = np.vstack((face,face2))
    #print(face)
    facemask = circle_mark(facemask,face,brw=1)

    #---------eyebrow-----------
    browl = dotsets[0][17:22]
    browr = dotsets[0][22:27]
    facemask = circle_mark(facemask,browl,brw=0)
    facemask = circle_mark(facemask,browr,brw=0)

    #----------eye--------------
    eyel = dotsets[0][36:42]
    eyer = dotsets[0][42:48]
    facemask = circle_mark(facemask,eyel,brw=0)
    facemask = circle_mark(facemask,eyer,brw=0)

    #---------mouth-------------
    mouth = dotsets[0][48:61]
    facemask = circle_mark(facemask,mouth,brw=0)

    #---------nose--------------
    #nose = np.vstack((dotsets[0][31:36],dotsets[0][42],dotsets[0][27],dotsets[0][39]))
    nose = np.vstack((dotsets[0][31:36],dotsets[0][29]))
    right = [dotsets[0][27][0]+1,dotsets[0][27][1]]
    left = [dotsets[0][27][0]-1,dotsets[0][27][1]]
    nose = np.vstack((dotsets[0][31:36],right,left))
    facemask = circle_mark(facemask,nose,brw=0)

    facemask = facemask.transpose()

    #facemask[5][15]=1
    # cv2.imshow("outImg",facemask)
    # #cv2.imshow("outImg",facemask)
    # cv2.waitKey(0)
    # num_space = np.sum(facemask).astype(int)
    # print(num_space)
    
    return facemask




def landmarks_68(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
    pic_array = np.array(img)
    h, w, d = pic_array.shape
    r,g,b = cv2.split(pic_array)
    img = cv2.merge([b, g, r])
    img = img.astype('uint8')
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   
    rects = detector(img_gray, 1)                      
    print(rects)
    # rawpots = np.array([[p.x, p.y] for p in predictor(img,rects[0]).parts()])
    if len(rects)==0:
        return []
    else:
        feature_points = np.array([[(p.x-w/2), (h/2-p.y)] for p in predictor(img,rects[0]).parts()])
    return feature_points


def sticker_spatial(vertices, img):
    w, h = img.shape[1],img.shape[0]
    store = np.zeros((h,w,2))
    for i in range(len(vertices)):
        x = int(np.round(vertices[i][0]))
        y = int(np.round(vertices[i][1]))
        x = min(x,w-1)
        y = min(y,h-1)
        store[y][x][0] = store[y][x][0] + vertices[i][2]
        store[y][x][1] = store[y][x][1] + 1
    store[:,:,1][np.where(store[:,:,1]==0)] = 1
    zstore = store[:,:,0]/store[:,:,1]

    return zstore

def generate_zstore(img):
    from face3d.morphable_model import MorphabelModel
    from face3d import mesh
    # --- 1. load model
    bfm = MorphabelModel('models/BFM/BFM.mat')
    #print('init bfm model success')
    img = np.array(img.cpu().squeeze(0).permute(1,2,0)*255)
    # --- 2. load fitted face
    feature_points = landmarks_68(img)
   
    if len(feature_points)==0:
        return []
    else:
        w,h = img.shape[1],img.shape[0]

        x = feature_points.copy()
        X_ind = bfm.kpt_ind # index of keypoints in 3DMM. fixed.
        sp, ep, s, angles, t3d = bfm.fit(x, X_ind, max_iter = 15)
        #print('s, angles, t = ',s, angles, t3d)

        # tp = bfm.get_tex_para('random')
        # colors1 = bfm.generate_colors(tp)

        # verify fitted parameters
        vertices = bfm.generate_vertices(sp, ep)
        transformed_vertices = bfm.transform(vertices, s, angles, t3d)

        image_vertices = mesh.transform.to_image(transformed_vertices, h, w)
        zstore = sticker_spatial(image_vertices, img)
    return zstore

def comp_arclen(a,c,x):
    def f(x):
        return (1 + ((x-c)*(2*a))**2)**0.5
    A = integrate.quad(f,0,x)[0]
    return A

def binary_equation(y1,z1,y2,z2):
    def flinear(x):# k:x[0], b:x[1]
        return np.array([x[0]*y1+x[1]-z1,x[0]*y2+x[1]-z2])
    yzlinear = fsolve(flinear,[0,0])
    return yzlinear

def solve_b(a,c,w,locate):
    '''
    find the upper limit of the integral 
    so that the arc length is equal to the width of the sticker
    locate: the highest point is on th right(1) b = -a*c^2, 
                                     left (2) b = -a*(upper-c)^2
    return: b , wn=upper(Width of converted picture)
    '''
    def func(x):
        def f(x):
            return (1 + ((x-c)*(2*a))**2)**0.5
        f = integrate.quad(f,0,x)[0]-w
        return f
    root = 0
    upper = fsolve(func,[root])[0] # The X coordinate when the arc length = w
    if(locate == 1):
        b = -a * (c**2)
    elif(locate == 2):
        b = -a * ((upper-c)**2)
    wn = int(np.floor(upper))
    return b, wn

def solve_a(hsegment,stride):
    '''
    solve 'a' according to Height drop in one step
    '''
    a = -hsegment/(stride)**2
    a = max(a, -1/stride)
    #print('a=',a,'hsegment=',hsegment,'stride=',stride)
    return a

def bilinear_interpolation(img,x,y):
    w,h = img.size
    xset = math.modf(x)
    yset = math.modf(y)
    u, v = xset[0], yset[0]
    x1, y1 = xset[1], yset[1]
    x2 = x1+1 if u>0 and x1+1<w else x1
    y2 = y1+1 if v>0 and y1+1<h else y1
    #x2, y2 = x1+1, y1+1
    p1_1 = np.array(img.getpixel((x1,y1)))
    p1_2 = np.array(img.getpixel((x1,y2)))
    p2_1 = np.array(img.getpixel((x2,y1)))
    p2_2 = np.array(img.getpixel((x2,y2)))

    pix = (1-u)*(1-v)*p1_1 + (1-u)*v*p1_2 + u*(1-v)*p2_1 + u*v*p2_2
    p = tuple(np.round(pix).astype(np.int32))
    return p


#def horizontal(sticker, zstore):
def horizontal(sticker,params):
    '''
    transform the picture according to parabola in horizontal direction
    input:
        sticker: Image type
        height: matrix (store height information for each coordinate)
    output:
        hor_sticker
    '''
    w, h = sticker.size
    c, hsegment, stride, locate = params[0],params[1],params[2],params[3]
    # c = 100
    # hsegment = 150
    # stride = c
    # locate = 1
        
    a = solve_a(hsegment,stride)
    b, wn = solve_b(a,c,w,locate)
    #print('a,b,c,wn,w = ',a,b,c,wn,w)
    
    top3 = np.ones((h,wn,3))*255
    top4 = np.zeros((h,wn,1))
    newimg = np.concatenate((top3,top4),axis=2)
    newimg = Image.fromarray(np.uint8(newimg))
    
    def f(x):
        return (1 + ((x-c)*(2*a))**2)**0.5
    x_arc = [integrate.quad(f,0,xnow+1)[0]  for xnow in range(wn)]
    z = np.zeros((1,wn))

    def zfunction(x):
        return a * ((x-c)**2) + b

    for i in range(wn):
        x_map =  min(x_arc[i],w-1)
        z[0][i] = zfunction(i)
        #print(x_map,jstart,int(jstart))
        for j in range(h):
            #y_map = j+jstart-np.floor(jstart)
            #y_map = j+np.modf(jstart)[0]
            y_map = j
            #print(j,jstart,np.floor(jstart),y_map)
            #print(x_map,y_map)
            pix = bilinear_interpolation(sticker,x_map,y_map)
            newimg.putpixel((i,j),pix)
            
    return newimg,z

def pitch(newimg,z,theta):
    #theta = math.radians(-10)
    w,h = newimg.size
    m = np.array([[1,0,0],
                [0,math.cos(theta),-math.sin(theta)],
                [0,math.sin(theta),math.cos(theta)]])
    invm = np.linalg.inv(m)
    
    x = np.array(range(w))
    y1, y2 = np.ones([1,w])*0, np.ones([1,w])*(h-1)
    first = np.vstack([x,y1,z]).T
    last = np.vstack([x,y2,z]).T
    pfirst = first.dot(m)
    plast = last.dot(m)
    #print(pfirst)
    
    #print(theta,m)
    hn = int(np.floor(np.max(plast[:,1])) - np.ceil(np.min(pfirst[:,1])))+1
    shifting = np.ceil(np.min(pfirst[:,1]))
    top3n = np.ones((hn,w,3))*255
    top4n = np.zeros((hn,w,1))
    endimg = np.concatenate((top3n,top4n),axis=2)
    endimg = Image.fromarray(np.uint8(endimg))

    # start = int(np.ceil(pfirst + shifting))
    # stop = int(np.floor(plast))
    start = np.ceil(pfirst[:,1] - shifting)
    stop = np.floor(plast[:,1] - shifting)

    for i in range(w):
        jstart = int(start[i])
        jstop = int(stop[i])
        def zconvert(y):
            parm = binary_equation(pfirst[i][1],pfirst[i][2],plast[i][1],plast[i][2])
            return parm[0]*y + parm[1]

        #print(x_map,jstart,int(jstart))
        for j in range(jstart,jstop+1):
            #print(jstart,jstop,shifting)
            raw_y = j+shifting
            raw_z = zconvert(raw_y)
            mapping = np.array([i,raw_y,raw_z]).dot(invm)
            #print(j,jstart,np.floor(jstart),y_map)
            #print(x_map,y_map)
            #print(mapping)
            pix = bilinear_interpolation(newimg,mapping[0],mapping[1])
            endimg.putpixel((i,j),pix)
    return endimg,shifting

def change_sticker(sticker,scale):
    new_weight = int(sticker.size[0]/scale)
    new_height = int(sticker.size[1]/scale)
    #print(new_weight,new_height)
    sticker = sticker.resize((new_weight,new_height),PIL.Image.Resampling.LANCZOS)
    return sticker


def deformation3d(sticker,operate_sticker,magnification,zstore,x,y):
    w, h = sticker.size
    
    area = zstore[y:y+h,x:x+w]
    #print('y,x=',y,x,'w,h=',w,h,area.shape)
    index = np.argmax(area)
    highesty = index // area.shape[1]   # Location coordinates of the highest point in the selected area
    highestx = index % area.shape[1]
    locate = 1 if highestx > area.shape[1]/2 else 2 # =1 if the highest point is to the right
    sign = 1 if highesty < area.shape[0]/2 else -1  # =1 if the highest point is on the top(Forward rotation)
    c = highestx
    if (locate==1):
        hsegment = area[highesty][highestx] - area[highesty][0]
        stride = c
    elif(locate==2):
        #hsegment = area[highesty][highestx] - area[highesty][w-1]
        hsegment = area[highesty][highestx] - area[highesty][area.shape[1]-1]
        stride = w - highestx
    
    #step = 10
    if (sign==1):
        step = max(min(20,area.shape[0]-highesty-1),1)
        #print('area.shape =',area.shape,'y,x=',highesty,highestx,'step=',step)
        partz = area[highesty][highestx] - area[highesty+step][highestx]
        party = step
        theta = min(math.atan(partz/party),math.radians(40))
        #theta = math.atan(partz/party)
    elif(sign==-1):
        step = max(min(20,highesty),1)
        partz = area[highesty][highestx] - area[highesty-step][highestx]
        party = step
        theta = max(-1 * math.atan(partz/party),math.radians(-40))
        #theta = -1 * math.atan(partz/party)
    operate_params = [c*magnification,hsegment,stride*magnification,locate]
    # print(operate_params)
    # print('theta = ',math.degrees(theta))
    newimg,z = horizontal(operate_sticker,operate_params)
    endimg,shifting = pitch(newimg,z,theta/2)
    #endimg.show()
    # if(sign==-1):
    #     y = y + int(shifting)
    #sticker = stick.transparent_back(endimg)
    #print(sticker.size)
    #sticker.show()
    sticker=change_sticker(endimg,magnification)

    return sticker,y



def get_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def rescale(img, scale, r32=False):
    if scale == 1.0: return img

    h = img.shape[0]
    w = img.shape[1]
    
    if r32:
        img = resize(img, (round_32(h * scale), round_32(w * scale)))
    else:
        img = resize(img, (int(h * scale), int(w * scale)))

    return img

def compute_composite_normals(img, msk, model, size):
    
    bin_msk = (msk > 0)

    bb = get_bbox(bin_msk)
    bb_h, bb_w = bb[1] - bb[0], bb[3] - bb[2]

    # create the crop around the object in the image to send through normal net
    img_crop = img[bb[0] : bb[1], bb[2] : bb[3], :]

    crop_scale = 1024 / max(bb_h, bb_w)
    img_crop = rescale(img_crop, crop_scale)
        
    # get normals of cropped and scaled object and resize back to original bbox size
    nrm_crop = get_omni_normals(model, img_crop)
    nrm_crop = resize(nrm_crop, (bb_h, bb_w))

    h, w, c = img.shape
    max_dim = max(h, w)
    if max_dim > size:
        scale = size / max_dim
    else:
        scale = 1.0
    
    # resize to the final output size as specified by input args
    out_img = rescale(img, scale, r32=True)
    out_msk = rescale(msk, scale, r32=True)
    out_bin_msk = (out_msk > 0)
    
    # compute normals for the entire composite image at it's output size
    out_nrm_bg = get_omni_normals(model, out_img)
    
    # now the image is at a new size so the parameters of the object crop change.
    # in order to overlay the normals, we need to resize the crop to this new size
    out_bb = get_bbox(out_bin_msk)
    bb_h, bb_w = out_bb[1] - out_bb[0], out_bb[3] - out_bb[2]
    
    # now resize the normals of the crop to this size, and put them in empty image
    out_nrm_crop = resize(nrm_crop, (bb_h, bb_w))
    out_nrm_fg = np.zeros_like(out_img)
    out_nrm_fg[out_bb[0] : out_bb[1], out_bb[2] : out_bb[3], :] = out_nrm_crop

    # combine bg and fg normals with mask alphas
    out_nrm = (out_nrm_fg * out_msk[:, :, None]) + (out_nrm_bg * (1.0 - out_msk[:, :, None]))
    return out_nrm





def harmonic_composition(bg_img,comp_img,mask_img,inference_size=256):
    
    print('loading depth model')
    dpt_model = create_depth_models()

    print('loading normals model')
    nrm_model = load_omni_model()

    print('loading intrinsic decomposition model')
    int_model = load_models('paper_weights')

    print('loading albedo model')
    alb_model = load_albedo_harmonizer()

    print('loading reshading model')
    shd_model = load_reshading_model('paper_weights')
    
    bg_img = cv2.resize(bg_img.squeeze(0).cpu().detach().numpy().transpose(1,2,0),(256,256))
    comp_img = cv2.resize(comp_img.squeeze(0).cpu().detach().numpy().transpose(1,2,0),(256,256))
    mask_img = cv2.resize(mask_img.cpu().detach().numpy().transpose(1,2,0),(256,256))
    

    # to ensure that normals are globally accurate we compute them at
    # a resolution of 512 pixels, so resize our shading and image to compute 
    # rescaled normals, then run the lighting model optimization
    bg_h, bg_w = bg_img.shape[:2]
    max_dim = max(bg_h, bg_w)
    scale = 512 / max_dim
    
    small_bg_img = rescale(bg_img, scale)
    small_bg_nrm = get_omni_normals(nrm_model, small_bg_img)
    
    result = run_pipeline(
        int_model,
        small_bg_img ** 2.2,
        resize_conf=0.0,
        maintain_size=True,
        linear=True
    )
    
    small_bg_shd = result['inv_shading'][:, :, None]
    
    
    coeffs, lgt_vis = get_light_coeffs(
        small_bg_shd[:, :, 0], 
        small_bg_nrm, 
        small_bg_img
    )

    # now we compute the normals of the entire composite image, we have some logic
    # to generate a detailed estimation of the foreground object by cropping and 
    # resizing, we then overlay that onto the normals of the whole scene
    comp_nrm = compute_composite_normals(comp_img, mask_img, nrm_model, inference_size)

    # now compute depth and intrinsics at a specific resolution for the composite image
    # if the image is already smaller than the specified resolution, leave it
    h, w, c = comp_img.shape
    
    max_dim = max(h, w)
    if max_dim > inference_size:
        scale = inference_size / max_dim
    else:
        scale = 1.0
    
    # resize to specified size and round to 32 for network inference
    img = rescale(comp_img, scale, r32=True)
    msk = rescale(mask_img, scale, r32=True)
    
    depth = get_depth(img, dpt_model)
    
    result = run_pipeline(
        int_model,
        img ** 2.2,
        resize_conf=0.0,
        maintain_size=True,
        linear=True
    )
    
    inv_shd = result['inv_shading']
    # inv_shd = rescale(inv_shd, scale, r32=True)

    # compute the harmonized albedo, and the subsequent color harmonized image
    alb_harm = harmonize_albedo(img, msk, inv_shd, alb_model, reproduce_paper=1) ** 2.2
    harm_img = alb_harm * uninvert(inv_shd)[:, :, None]

    # run the reshading model using the various composited components,
    # and our lighting coefficients computed from the background
    comp_result = compute_reshading(
        harm_img,
        msk,
        inv_shd,
        depth,
        comp_nrm,
        alb_harm,
        coeffs,
        shd_model
    )
    img = torch.from_numpy(comp_result['composite'].transpose(2,0,1)).unsqueeze(0)
    img = F.interpolate(img,(128,128))
    return img
    

from face_detector import YoloDetector, YOLOv8_face
from retinaface.pre_trained_models import get_model
from insightface.app import FaceAnalysis
from models.mtcnn import FaceDetector
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from models.MogFace.core.workspace import register, create, global_config, load_config
def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model

def load_thermal_detector(config,device):
    if config.thermal_detector =='yolov5':
        model = load_model('weights/yolov5n_face.pt', device)
    elif config.thermal_detector =='yolov8':
        model =  YOLOv8_face('weights/yolov8n-face.onnx', conf_thres=config.thermal_threshold, iou_thres=0.4)
    elif  config.thermal_detector =='retinaface':
        model = get_model("resnet50_2020-07-20", max_size=416,device=device)
    elif  config.thermal_detector =='dlib':
        model = dlib.cnn_face_detection_model_v1('models/mmod_human_face_detector.dat')
    elif config.thermal_detector =='opencv_dnn':
        model = cv2.dnn.readNetFromTensorflow('models/opencv_face_detector_uint8.pb','models/opencv_face_detector.pbtxt')
    elif  config.thermal_detector =='scrfd':
        model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        model.prepare(ctx_id=0, det_size=(640, 640))
    elif  config.thermal_detector =='mtcnn':
        model  = FaceDetector()
    elif config.thermal_detector =="dsfd":
        model = SSD("test")
        model.load_state_dict(torch.load('models/DSFD/weights/WIDERFace_DSFD_RES152.pth'))
        model.to(device).eval()
    elif config.thermal_detector =="ulfd":
        model = pipeline(Tasks.face_detection, 'damo/cv_manual_face-detection_ulfd')
    elif config.thermal_detector =="mogface":
        #self.model = pipeline(Tasks.face_detection, 'damo/cv_resnet101_face-detection_cvpr22papermogface')
        config = 'models/MogFace/configs/mogface/MogFace_E.yml'
        # generate det_info and det_result
        cfg = load_config(config)
        cfg['phase'] = 'test'
        if 'use_hcam' in cfg and cfg['use_hcam']:
            # test_th
            cfg['fp_th'] = 0.12
        model = create(cfg.architecture)
        model.load_state_dict(torch.load('models/MogFace/configs/mogface/model_140000.pth'))
        model.to(device)
    
    return model
    
    
def load_visual_detector(config,device):
    if config.visual_detector =='yolov5':
        model = YoloDetector(target_size=720, device=device, min_face=90)
    elif config.visual_detector =='yolov8':
        model = load_model('weights/yolov8n_face.pt', device)
    elif  config.visual_detector =='retinaface':
        model = get_model("resnet50_2020-07-20", max_size=416,device=device)
    elif  config.visual_detector =='dlib':
        model = dlib.cnn_face_detection_model_v1('models/mmod_human_face_detector.dat')
    elif config.visual_detector =='opencv_dnn':
        model = cv2.dnn.readNetFromTensorflow('models/opencv_face_detector_uint8.pb','models/opencv_face_detector.pbtxt')
    elif  config.visual_detector =='scrfd':
        model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        model.prepare(ctx_id=0, det_size=(640, 640))
    elif  config.visual_detector =='mtcnn':
        model  = FaceDetector()
    elif config.visual_detector =="dsfd":
        model = SSD("test")
        model.load_state_dict(torch.load('models/DSFD/weights/WIDERFace_DSFD_RES152.pth'))
        model.to(device).eval()
    elif config.visual_detector =="ulfd":
        model = pipeline(Tasks.face_detection, 'damo/cv_manual_face-detection_ulfd')
    elif config.visual_detector =="mogface":
        #self.model = pipeline(Tasks.face_detection, 'damo/cv_resnet101_face-detection_cvpr22papermogface')
        config = 'models/MogFace/configs/mogface/MogFace_E.yml'
        # generate det_info and det_result
        cfg = load_config(config)
        cfg['phase'] = 'test'
        if 'use_hcam' in cfg and cfg['use_hcam']:
            # test_th
            cfg['fp_th'] = 0.12
        model = create(cfg.architecture)
        model.load_state_dict(torch.load('models/MogFace/configs/mogface/model_140000.pth'))
        model.to(device)
    
    return model
    
    
    
    
def visual_detection(config,visual_model,img,device):
    if config.visual_detector == 'yolov5':   
        output = visual_model.detector(img)[0] # (b,:,14)
        score = output[...,4]
        #score = torch.sigmoid(score)
        max_score = torch.max(score,dim=1)
        
        det_loss = torch.mean(max_score.values)   
        
    if config.visual_detector == 'yolov8':   

        output = visual_model(img)[0]
        
        score = output[:,4,:]
        #score = torch.sigmoid(score)
        max_score = torch.max(score,dim=1)
        det_loss = max_score.values 
        
    elif config.visual_detector == 'retinaface':
        #sub_img = img.detach().cpu().squeeze(0).numpy().transpose(1,2,0)
        det_loss = visual_model.predict_jsons(img)
        det_loss = det_loss.to(device)
    elif config.visual_detector == 'dlib':
        sub_img = img.detach().cpu().squeeze(0).numpy().transpose(1,2,0)*255
        sub_img=sub_img.astype(np.uint8)
        sub_img = imutils.resize(sub_img, width=300)
        # convert the image to grayscale
        sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGRA2GRAY)
        # detect faces in the image 
        rects = visual_model(sub_img, upsample_num_times=1)    
        if len(rects) !=0:
            rect = rects[0]
            confidence = rect.confidence
            det_loss = torch.tensor(confidence)
            det_loss = det_loss.to(device)
        else:
            det_loss =  0
    elif config.visual_detector =='opencv_dnn':
        sub_img = img.detach().cpu().squeeze(0).numpy().transpose(1,2,0)*255
        sub_img=sub_img.astype(np.uint8)
        blob = cv2.dnn.blobFromImage(sub_img,1.0,(112,112),[104,117,123],False,False)

        visual_model.setInput(blob)
        detections = visual_model.forward()
        bboxes = []
        ret = 0 
        max_score = torch.from_numpy(detections[:,:,:,2])
        #max_score = torch.sigmoid(max_score)
        det_loss = torch.max(max_score).to(device)
    elif config.visual_detector =='scrfd':
        sub_img = img.detach().cpu().squeeze(0).numpy().transpose(1,2,0)*255
        sub_img=sub_img.astype(np.uint8)
        score = visual_model.get(sub_img)
        if len(score)!=0:
            det_loss = torch.max(torch.from_numpy(np.array(score))).to(device)
        else:
            det_loss = 0
        

    elif config.visual_detector =='mtcnn':
        sub_img = transforms.ToPILImage()(img.squeeze(0)).convert('RGB')#img_batch_applied.cpu().squeeze(0).numpy().transpose(1,2,0)*255
        bboxes, landmarks = visual_model.detect(sub_img)    
        if len(bboxes)!=0:
            score = bboxes[:,-1]
            det_loss = torch.max(torch.from_numpy(np.array(score))).to(device)
        else:
            det_loss = 0
        

    elif config.visual_detector =="dsfd":
        
        sub_img = img.cpu().squeeze(0).numpy().transpose(1,2,0)*255
        sub_img = sub_img.astype(np.uint8)
        detections = visual_model.detect_on_image(sub_img, (128,128), device, is_pad=False, keep_thresh=0.1)
        score = detections[:,-1]
        if len(score)!=0:
            det_loss = torch.max(torch.from_numpy(np.array(score))).to(device)
        else:
            det_loss = 0

    elif config.visual_detector =="ulfd":
        sub_img = img.detach().cpu().squeeze(0).numpy().transpose(1,2,0)*255
        sub_img = sub_img.astype(np.uint8)
        result = visual_model(sub_img)
        scores = result['scores']
        if len(scores)!=0:
            det_loss = torch.max(torch.from_numpy(np.array(scores))).to(device)
        else:
            det_loss = 0
    elif config.visual_detector =="mogface":
        
        img = F.interpolate(img,(416,416)).to(device)
        
        result = visual_model(img)
        scores = torch.max(result[0].squeeze(0))
        det_loss = scores
        img = F.interpolate(img,(128,128)).to(device)
        '''sub_img_batch_applied = img_batch_applied.cpu().squeeze(0).numpy().transpose(1,2,0)*255
        sub_img_batch_applied = sub_img_batch_applied.astype(np.uint8)
        result = self.model(sub_img_batch_applied)
        print('result=',result)
        scores = result['scores']
        if len(scores)!=0:
            det_loss = torch.max(torch.from_numpy(np.array(scores))).to(device)
        else:
            det_loss = 0'''
    
    #all_det_loss[iter] = det_loss
    
    #max_det_loss = torch.max(all_det_loss)
        
    '''tv_loss = self.total_variation(adv_patch,self.mask) * 0.01
    
    if adv_patch.shape[1]==4:
        nps = self.nps_calculator(adv_patch[:,:3].to(device),adv_patch.shape[-1])
    else:
        nps = self.nps_calculator(adv_patch.to(device),adv_patch.shape[-1])
    nps = nps*0.1
    tv_loss = tv_loss.cpu()'''
    #nps = nps.cpu()
    
    loss = det_loss#+tv_loss[0]+nps[0]
    return loss
    
    
def thermal_detection(config,model,img_batch,device):
    if config.thermal_detector == 'yolov5':
        '''transform = transforms.Compose([transforms.Resize((128,128)),
                                               transforms.ToTensor()])
        img_path = 'processing_images/img_batch_thermal.jpg'
        ###thermal
        #img_path = 'datasets/image4.jpg'
        image = Image.open(img_path)#.convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        save_img_sub = image.squeeze(0).cpu().detach().numpy().transpose(1,2,0)
        cv2.imwrite('processing_images/img_batch_thermal.jpg',save_img_sub[:,:,::-1]*255)'''
        out, train_out = model(img_batch)
        obj_confidence = out[:, :, 4]
        max_obj_confidence, _ = torch.max(obj_confidence, dim=1)
        obj_loss = torch.mean(max_obj_confidence)

    elif config.thermal_detector == 'yolov8':   
        sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
        sub_img_batch_applied=sub_img_batch_applied.astype(np.uint8)      
        score= model.detect(sub_img_batch_applied) 
        
        score = torch.from_numpy(np.array(score))
        #score = torch.sigmoid(score)
        max_score = torch.max(score)
        obj_loss = torch.mean(max_score) 
    elif config.thermal_detector == 'retinaface':
        #sub_img_batch_applied = p_img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)
        
        det_loss = model.predict_jsons(img_batch)
        obj_loss = det_loss.to(device)
    elif config.thermal_detector == 'dlib':
        sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
        sub_img_batch_applied=sub_img_batch_applied.astype(np.uint8)
        sub_img_batch_applied = imutils.resize(sub_img_batch_applied, width=300)
        # convert the image to grayscale
        sub_img_batch_applied = cv2.cvtColor(sub_img_batch_applied, cv2.COLOR_BGRA2GRAY)
        # detect faces in the image 
        rects = model(sub_img_batch_applied, upsample_num_times=1)    
        if len(rects) !=0:
            rect = rects[0]
            confidence = rect.confidence
            det_loss = torch.tensor(confidence)
            obj_loss = det_loss.to(device)
        else:
            obj_loss =  0
    elif config.thermal_detector =='opencv_dnn':
        sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
        sub_img_batch_applied=sub_img_batch_applied.astype(np.uint8)
        blob = cv2.dnn.blobFromImage(sub_img_batch_applied,1.0,(112,112),[104,117,123],False,False)
        frameHeight = sub_img_batch_applied.shape[0]
        frameWidth = sub_img_batch_applied.shape[1]
        model.setInput(blob)
        detections = model.forward()
        bboxes = []
        ret = 0 
        max_score = torch.from_numpy(detections[:,:,:,2])
        #max_score = torch.sigmoid(max_score)
        obj_loss = torch.max(max_score).to(device)
    elif config.thermal_detector =='scrfd':
        sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
        sub_img_batch_applied=sub_img_batch_applied.astype(np.uint8)
        score = model.get(sub_img_batch_applied)
        if len(score)!=0:
            obj_loss = torch.max(torch.from_numpy(np.array(score))).to(device)
        else:
            obj_loss = 0
    elif config.thermal_detector =='mtcnn':
        sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
        sub_img_batch_applied = sub_img_batch_applied.astype(np.uint8)
        sub_img_batch_applied = Image.fromarray(sub_img_batch_applied)
        bboxes, landmarks = model.detect(sub_img_batch_applied)
        
        if len(bboxes) == 0:
            obj_loss = 0
        else:
            score = bboxes[:,-1]
            obj_loss = torch.max(torch.from_numpy(np.array(score))).to(device)
        '''
        if len(score)!=0:
            det_loss = torch.max(torch.from_numpy(np.array(score))).to(device)
        else:
            det_loss = 0'''
        

    elif config.thermal_detector =="dsfd":
        target_size = (128, 128)
        sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
        sub_img_batch_applied = sub_img_batch_applied.astype(np.uint8)
        detections = model.detect_on_image(sub_img_batch_applied, target_size, device, is_pad=False, keep_thresh=0.1)
        score = detections[:,-1]
        if len(score)!=0:
            obj_loss = torch.max(torch.from_numpy(np.array(score))).to(device)
        else:
            obj_loss = 0
            
    
        
    elif config.thermal_detector =="ulfd":
        sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
        sub_img_batch_applied = sub_img_batch_applied.astype(np.uint8)
        result = model(sub_img_batch_applied)
        scores = result['scores']
        if len(scores)!=0:
            obj_loss = torch.max(torch.from_numpy(np.array(scores))).to(device)
        else:
            obj_loss = 0
    elif config.thermal_detector =="mogface":
        p_img_batch = F.interpolate(img_batch,(256,256))
        result = model(p_img_batch)
        scores = torch.max(result[0].squeeze(0))
        obj_loss = scores
        
    return obj_loss





def augment_patch(img, mask_src,contrast_var, brightness_var,noise_factor,device):

        min_contrast = 1 - contrast_var
        max_contrast = 1 + contrast_var
        min_brightness = -brightness_var
        max_brightness = brightness_var
        contrast = get_random_tensor(img, min_contrast, max_contrast,device) * mask_src
        brightness = get_random_tensor(img, min_brightness, max_brightness,device) * mask_src
        noise = torch.empty(img.shape, device=device).uniform_(-1, 1) * noise_factor * mask_src
        img = img * (1-mask_src)+(img * contrast + brightness + noise)*mask_src
        return img

def get_random_tensor(adv_patch, min_val, max_val,device):
    t = torch.empty(adv_patch.shape[0], device=device).uniform_(min_val, max_val)
    t = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    t = t.expand(-1, adv_patch.size(-3), adv_patch.size(-2), adv_patch.size(-1))
    return t


def aug_sticker(location):
    random_number1 = random.uniform(-2, 2)
    location[0] = location[0] + random_number1
    random_number2 = random.uniform(-10, 10)
    location[1] = location[1] + random_number2
    
    return location

from libcom import ImageHarmonizationModel
from libcom import color_transfer

def image_harmonic(comp_img1,comp_mask1):
    
    comp_img1 = comp_img1.detach().cpu().squeeze(0).numpy().transpose(1,2,0)*255
    comp_img1 = comp_img1[:,:,::-1].astype(np.uint8)
    comp_mask1 = comp_mask1.detach().cpu().numpy().transpose(1,2,0)*255
    comp_mask1 = comp_mask1[:,:,0].astype(np.uint8)

    

    trans_comp = reinhard(comp_img1, comp_mask1)
    

    

    CDTNet = ImageHarmonizationModel(device='0', model_type='CDTNet')
    
    CDT_result1 = CDTNet(trans_comp, comp_mask1)

    img = torch.from_numpy(CDT_result1).permute(2,0,1).unsqueeze(0)/255
    return img

def reinhard(comp_img, comp_mask):
    comp_mask = np.where(comp_mask > 127, 255, 0).astype(np.uint8)
    comp_lab  = cv2.cvtColor(comp_img, cv2.COLOR_BGR2Lab)
    bg_mean, bg_std = cv2.meanStdDev(comp_lab, mask=255 - comp_mask)
    fg_mean, fg_std = cv2.meanStdDev(comp_lab, mask=comp_mask)
    ratio = (bg_std / fg_std).reshape(-1)
    offset = (bg_mean - fg_mean * bg_std / fg_std).reshape(-1)
    trans_lab = cv2.convertScaleAbs(comp_lab * ratio + offset)
    trans_img = cv2.cvtColor(trans_lab, cv2.COLOR_Lab2BGR)
    trans_comp = np.where(comp_mask[:,:,np.newaxis] > 127, trans_img, comp_img)
    return trans_comp










def get_gx(tensor,device):
    B, C, W, H = tensor.shape
    sobelx = torch.tensor([[1,0,-1], [2,0,-2], [1,0,-1]], device=device).float()
    sobely = sobelx.T

    G_x = F.conv2d(tensor.view(-1,1, W, H), sobelx.view((1,1,3,3)),padding=1).view(B, C, W, H)
    G_y = F.conv2d(tensor.view(-1,1, W, H), sobely.view((1,1,3,3)),padding=1).view(B, C, W, H)

    G = torch.sqrt(torch.pow(G_x, 2)+torch.pow(G_y, 2)).float()

    gx = (G-G.min())/(G.max()-G.min())
    return gx

def get_gradient(tensor, block_size=15, overlap=5, threshold=0):
    
    stride = block_size - overlap
    padding = []
    for v in tensor.shape[-2:]:
        left = v % stride
        if left <= overlap:
            need = overlap
        else:
            need = block_size
        pad = need - left
        padding.append(int(pad / 2))
        padding.append(pad - padding[-1])

    tensor = F.pad(tensor, tuple(padding))

    B, C, W, H = tensor.shape

    patches = tensor.unfold(2, block_size, stride).unfold(3, block_size, stride)
    mask = ((patches.sum([1,-2,-1]) / torch.prod(torch.tensor(patches.shape)[[1,-2,-1]])) >= threshold).float().unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

    def fold(x):
        x = x * mask
        x = x.contiguous().view(B, C, -1, block_size * block_size).permute(0, 1, 3, 2)
        x = x.contiguous().view(B, C*block_size*block_size, -1)
        return F.fold(x, output_size=(H,W), kernel_size=block_size, stride=stride)

    ox = torch.ones_like(patches)
    r =  fold(patches) / fold(ox)
    r = F.pad(r, tuple(-i for i in padding))
    r[r != r] = 0
    return r
    
def apply_gradient(img, grad, smooth_factor=2.3):
    t = 1 - grad * smooth_factor
    clipped = torch.clip(t, 0, 1)
    return torch.mul(img, clipped)

def lgs(tensor,device, threshold=0.2, smooth_factor=2.3, block_size=15, overlap=5):
    return apply_gradient(tensor, get_gradient(get_gx(tensor,device), block_size=block_size, overlap=overlap, threshold=threshold), smooth_factor=smooth_factor)

