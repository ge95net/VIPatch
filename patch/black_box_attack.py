import sys
import os
sys.path.append("/home/qiuchengyu/AdversarialMask") 
import random
from pathlib import Path
import pickle

import cv2
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
import function
from config import patch_config_types
from nn_modules import LandmarkExtractor, FaceXZooProjector, TotalVariation,NPSCalculator
from function import *
#from utils.general import xywh2xyxy


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
from face_detector import YoloDetector, YOLOv8_face
from retinaface.pre_trained_models import get_model
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from models.mtcnn import FaceDetector
from models.DSFD.face_ssd_infer import SSD
import dlib
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import argparse
from dataloader import *
import warnings
from models.experimental import attempt_load
warnings.simplefilter('ignore', UserWarning)

global device
#os.environ['CUDA_VISIBLE_DEVICEs'] = '7'
cuda_num = '4'
device = torch.device("cuda:"+cuda_num if torch.cuda.is_available() else "cpu")
#os.environ['CUDA_LAUNCH_BLOCKING'] = "3"
print('device is {}'.format(device), flush=True)
from models.MogFace.core.workspace import register, create, global_config, load_config
def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model
def set_random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
class AdversarialMask:
    def __init__(self, config):
        self.config = config
        set_random_seed(seed_value=self.config.seed)
        self.data_detector = YoloDetector(device=device, min_face=90)
        self.train_loader = function.get_train_loaders(self.config,self.data_detector)
        #self.thermal_loader = image_Testdata('datasets/thermal/gray/train/images',self.config)



        face_landmark_detector = function.get_landmark_detector(self.config, device)
        self.location_extractor = LandmarkExtractor(device, face_landmark_detector, self.config.img_size).to(device)
        self.fxz_projector = FaceXZooProjector(device, self.config.img_size, self.config.patch_size,self.config).to(device)
        self.total_variation = TotalVariation(device).to(device)
        
        self.nps_calculator = NPSCalculator('non_printability/30values.txt', config.patch_size[0],device=device)
        
        self.retinaface_transform = A.Compose([A.LongestMaxSize(max_size=960, p=1), A.Normalize(p=1)])

        self.device = device
        self.best_patch = None
        self.visual_model = load_visual_detector(self.config,device)
        self.thermal_model = load_thermal_detector(self.config,device)

    def train(self):
        with torch.no_grad():
            if self.config.mode=='train':
                adv_patch_cpu = function.load_mask(self.config,'data/masks/blue.png',device)
                adv_patch_hsv = adv_patch_cpu.clone()
                
                adv_patch_zeros = torch.zeros_like(adv_patch_cpu)
                v_max = 0
                v_min = 9999

                        
                
                #adv_patch_cpu[:,:3,:,:] = torch.zeros((1, 3, self.config.patch_size[0], self.config.patch_size[1]), dtype=torch.float32) +1
                mask = adv_patch_cpu[:,3].unsqueeze(0)
                self.mask = mask
                
                #self.v_boundary = [v_min.cpu().numpy(),v_max.cpu().numpy()]
                
                white_patch = torch.zeros((1, 3, self.config.patch_size[0], self.config.patch_size[1]), dtype=torch.float32) +1
                white_patch = white_patch.to(device)
                
                white_patch = torch.cat((white_patch,mask),1)
                #adv_patch_cpu[:,:3,:,:] = torch.rand((1, 3, self.config.patch_size[0], self.config.patch_size[1]), dtype=torch.float32)
                #adv_patch_cpu = torch.clamp(adv_patch_cpu, min=0.1, max=0.9)
            else:
                adv_patch_cpu = function.load_mask(self.config,'data/masks/blue.png',device)
            
            #adv_patch_cpu = F.interpolate(adv_patch_cpu,(416,416))
            threshold = 0.83
            total_case = 0
            success_case = 0
            quries_list = []
            for i, (visual_img,thermal_img) in enumerate(self.train_loader):
 
                visual_img = visual_img.to(device)
                thermal_img = thermal_img.to(device)
                if self.config.mode == 'test_defense':
                    if self.config.defense_mode == 'lgs':
                        threshold=0.1
                        smooth_factor=3.3
                        x = lgs(visual_img,device, threshold=threshold, smooth_factor=smooth_factor)
                        loss,image = self.forward_step_sticker(x,adv_patch_cpu)
                    if self.config.defense_mode == 'JPEG':
                        from io import BytesIO
                        image_array = visual_img.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                        image_pil = Image.fromarray((image_array * 255).astype('uint8'))
                        image_pil.save('save_img.jpg', format='JPEG', optimize=True, quality=85)
                        output_stream = cv2.imread('save_img.jpg')
                        
                        compressed_image_data = cv2.cvtColor(output_stream, cv2.COLOR_BGR2RGB)
            
                        compressed_image_data = torch.from_numpy(compressed_image_data).unsqueeze(0).permute(0,3,1,2).float().to(device)/255
                        loss,image = self.forward_step_sticker(compressed_image_data,adv_patch_cpu)
                    if self.config.defense_mode == 'median_filter':
                        from scipy.ndimage import median_filter, gaussian_filter
                        visual_img = visual_img.squeeze(0).cpu().numpy().transpose(1,2,0)
                        
                        filtered_image_np = median_filter(visual_img, size=3)
                        filtered_image_np = torch.from_numpy(filtered_image_np).permute(2,0,1).unsqueeze(0).to(device)
                        loss,image = self.forward_step_sticker(filtered_image_np,adv_patch_cpu)
                        
                    if self.config.defense_mode == 'gaussian_filter':
                        from scipy.ndimage import median_filter, gaussian_filter
                        visual_img = visual_img.squeeze(0).cpu().numpy().transpose(1,2,0)
                        filtered_image_np = gaussian_filter(visual_img, sigma=1, mode='reflect')
                        filtered_image_np = torch.from_numpy(filtered_image_np).permute(2,0,1).unsqueeze(0).to(device)
                        loss,image = self.forward_step_sticker(filtered_image_np,adv_patch_cpu)
                    if self.config.defense_mode == 'bit_depth_reduction':
                        num_bits = 4
                        depth_range = 2 ** num_bits - 1
                        quantized_image = torch.round(visual_img * depth_range) / depth_range
                        loss,image = self.forward_step_sticker(quantized_image,adv_patch_cpu)
                        
                    if loss < self.config.visual_threshold:
                        success_case+=1
                    total_case+=1
                    asr = success_case/total_case
                    #if loss > self.config.threshold:
                    print('iteration=%d,asr=%f,loss=%f'%(i,asr,loss))
                
                if self.config.mode =='test_visual':
                    
                    visual_img = F.interpolate(visual_img,(416,416))
                    loss,image = self.forward_step_sticker(visual_img,adv_patch_cpu)
                    
       
                    
                    if loss < self.config.visual_threshold:
                        success_case+=1
                    total_case+=1
                    asr = success_case/total_case
                    #if loss > self.config.threshold:
                    print('iteration=%d,asr=%f,loss=%f'%(i,asr,loss))
                if self.config.mode =='test_thermal':
                    thermal_img = F.interpolate(thermal_img,(416,416))
                    loss,image = self.forward_step_thermal(thermal_img,aug=False)
                    
                    save_img = image.squeeze(0).permute(1,2,0).cpu().numpy()
                    
                    
                    
                    
                    if loss < self.config.thermal_threshold:
                        success_case+=1
                    total_case+=1
                    asr = success_case/total_case
                    #if loss > self.config.threshold:
                    print('iteration=%d,asr=%f,loss=%f'%(i,asr,loss))
                
                elif self.config.mode =='train':
                    batch = visual_img.shape[0]
                    if i == 0:
                        patches = adv_patch_cpu.repeat(batch,1,1,1)

                    image_size = [128,128]
                  
                    
                  
                    patches,loss_visual_min,loss_thermal_min,best_img = stickers(forward_step=self.forward_step,forward_step_sticker=self.forward_step_sticker,forward_step_thermal = self.forward_step_thermal,img_batch=visual_img,thermal_image = thermal_img,white_patch=white_patch,patches=patches,\
                        adv_patch_hsv=adv_patch_hsv,n_queries=self.config.n_queries,image_size=image_size,applied_mask=mask,device=device,visual_threshold=self.config.visual_threshold,thermal_threshold=self.config.thermal_threshold,do_aug=self.config.do_aug,harmonic = self.config.harmonic_color,model = self.config.thermal_detector)
                    
                   
                    if loss_visual_min < self.config.visual_threshold and loss_thermal_min < self.config.thermal_threshold :
                        success_case += 1
                    total_case+=1
                    print('visual_model=%s,thermal_model=%s,success_case=%i,total_case=%i,ASR=%f'%(str(self.config.visual_detector),str(self.config.thermal_detector),success_case,total_case,success_case/total_case))
                    
          
    def forward_step(self, img_batch, adv_patch_cpu,image_size,numnumnum=-1,uv_mask = None,first=True,get_mask=False):
       
        img_batch = img_batch.to(device)
        img_batch = img_batch.expand(adv_patch_cpu.shape[0],-1,-1,-1)
        adv_patch = adv_patch_cpu.to(device)
        
        if first == True:
            img_batch = F.interpolate(img_batch,(image_size[0],image_size[1]))
            adv_patch_cpu = F.interpolate(adv_patch_cpu,(image_size[0],image_size[1]))
            adv_patch = F.interpolate(adv_patch,(image_size[0],image_size[1]))
            img_batch_applied,rgb_mask = self.applied_mask(img_batch,adv_patch_cpu,first=True,get_mask=get_mask)
            img_batch_applied = img_batch_applied.to(device)
            if get_mask == True:
                self.rgb_mask = rgb_mask
                self.rgb_mask = self.rgb_mask.to(device)

        else:

            img_batch_applied = img_batch*(1 -self.rgb_mask ) + adv_patch * self.rgb_mask

        return_img = img_batch_applied
        
        
        
        
        img_batch_applied = torch.clamp(img_batch_applied, 0, 1)


        loss = visual_detection(self.config,self.visual_model,img_batch_applied,device)
        #### find the original face box
        
        
        return loss,return_img,adv_patch,self.rgb_mask
    

    def forward_step_origin(self, img_batch,adv_patch_cpu,image_size,mask,numnumnum=-1,uv_mask = None,first=True):
        if mask is not None:
            self.mask = mask
        img_batch = img_batch.to(device)
        img_batch = img_batch.expand(adv_patch_cpu.shape[0],-1,-1,-1)
        adv_patch = adv_patch_cpu.to(device)
        mask = mask.to(device).unsqueeze(0)
        if first == True:
            self.mask = None
            img_batch = F.interpolate(img_batch,(image_size[0],image_size[1]))
            adv_patch_cpu = F.interpolate(adv_patch_cpu,(image_size[0],image_size[1]))
            adv_patch = F.interpolate(adv_patch,(image_size[0],image_size[1]))
            #img_batch_applied,self.mask = self.applied_mask(img_batch,adv_patch_cpu)
            #save_mask = self.mask.cpu().numpy().transpose(1,2,0)
            adv_patch[:,0,:,:] = 0.76078
            adv_patch[:,1,:,:] = 0.54118
            adv_patch[:,2,:,:] = 0.5451
            
        else:
            img_batch = F.interpolate(img_batch,(image_size[0],image_size[1]))
            img_batch_applied = img_batch
            if self.config.mode == 'test':
                self.mask = adv_patch_cpu.clone()
                self.mask[self.mask!=0] = 1
                self.mask = self.mask[:,0,:,:]
        adv_patch = adv_patch[:,:3,:,:]
        

        img_batch_applied = img_batch * (1-mask) + adv_patch_cpu*mask

        loss = visual_detection(self.config,self.visual_model,img_batch_applied,device)
           
        
        return loss,self.mask,img_batch_applied,adv_patch
    

    def forward_step_sticker(self, img_batch,adv_patch,mode='origin'):
        
        if mode == 'paste':
            adv_patch = torch.cat((adv_patch,self.mask),1)
    
            img_batch_applied,_ = self.applied_mask(img_batch,adv_patch,first=False)

        else:
            
            img_batch_applied = img_batch
        
        loss = visual_detection(self.config,self.visual_model,img_batch_applied,device)
        
        return loss,img_batch_applied
    
    
    def forward_step_thermal(self, img_batch,aug=True):
        
        if aug==True:
            img_batch = augment_patch(img = img_batch, mask_src=self.rgb_mask,contrast_var=0.2,brightness_var=0.1,noise_factor=0.05,device=device)
        loss = thermal_detection(self.config,self.thermal_model,img_batch,device)
        
        return loss,img_batch
    
    def applied_mask(self, img_batch,adv_patch_cpu,first,uv_mask_src=None,get_mask=True):
        preds = self.location_extractor(img_batch) # 68 landmarks

        #####load mask
        if adv_patch_cpu.shape[1]==4:
            if uv_mask_src == None:
                adv_patch = adv_patch_cpu[:,:3]
                uv_mask_src = adv_patch_cpu[:,3]
                uv_mask_src[uv_mask_src<0.5] = 0
                uv_mask_src[uv_mask_src>=0.5] = 1
            else:
                uv_mask_src = uv_mask_src
            
            img_batch_applied,mask = self.fxz_projector(img_batch, preds, adv_patch, uv_mask_src=uv_mask_src,first=first,is_3d=True,do_aug=self.config.do_aug,get_mask=get_mask)
        else:
        ####random generate mask
            if uv_mask_src == None:
                adv_patch = adv_patch_cpu[:,:3]
                uv_mask_src = adv_patch_cpu[:,3]
                uv_mask_src[uv_mask_src<0.5] = 0
                uv_mask_src[uv_mask_src>=0.5] = 1
            else:
                uv_mask_src = uv_mask_src
            img_batch_applied,mask = self.fxz_projector(img_batch, preds, adv_patch_cpu,first, do_aug=self.config.do_aug,uv_mask_src=uv_mask_src,get_mask=get_mask)
        return img_batch_applied,mask
    


    

    def apply_random_grid_sample(self, face_mask):
        theta = torch.zeros((face_mask.shape[0], 2, 3), dtype=torch.float, device=self.device)
        rand_angle = torch.empty(face_mask.shape[0], device=self.device).uniform_(self.minangle, self.maxangle)
        theta[:, 0, 0] = torch.cos(rand_angle)
        theta[:, 0, 1] = -torch.sin(rand_angle)
        theta[:, 1, 1] = torch.cos(rand_angle)
        theta[:, 1, 0] = torch.sin(rand_angle)
        theta[:, 0, 2].uniform_(self.min_trans_x, self.max_trans_x)  # move x
        theta[:, 1, 2].uniform_(self.min_trans_y, self.max_trans_y)  # move y
        grid = F.affine_grid(theta, list(face_mask.size()))
        augmented = F.grid_sample(face_mask, grid, padding_mode='reflection')
        return augmented
   

def main():
    mode = 'universal'
    config = patch_config_types[mode]()
    print('Starting train...', flush=True)
    adv_mask = AdversarialMask(config)
    adv_mask.train()
    print('Finished train...', flush=True)


if __name__ == '__main__':
    main()