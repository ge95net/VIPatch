import sys

import os
from imutils import face_utils
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
import dlib
from function import *
import function
from models.experimental import attempt_load
from config import patch_config_types
from nn_modules import LandmarkExtractor, FaceXZooProjector, TotalVariation,NPSCalculator

from models.utils.general import xywh2xyxy
from face_detector import YoloDetector
from dataloader import *
import warnings
from models.utils.general import check_img_size, non_max_suppression_face
import kornia

from face_detector import YoloDetector, YOLOv8_face
from retinaface.pre_trained_models import get_model
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from models.mtcnn import FaceDetector
from models.DSFD.face_ssd_infer import SSD
import dlib
from modelscope.pipelines import pipeline
from models.MogFace.core.workspace import register, create, global_config, load_config
from modelscope.utils.constant import Tasks
warnings.simplefilter('ignore', UserWarning)

global device
os.environ['CUDA_VISIBLE_DEVICEs'] = '4'
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
print('device is {}'.format(device), flush=True)





class InfraredBlock():
    def __init__(self,L,box):
       
        x = random.randint(int(box[0]),int(box[2]))
        y = random.randint(int(box[1]),int(box[3]))
        if y == box[3]:
            y = box[3]-1
        if x == box[2]:
            x = box[2]-1
        if y == box[1]:
            y = box[1]+1
        if x == box[0]:
            x = box[0]+1
        
        '''while mask[y][x] == 0:
            x = random.randint(box[0][0],box[0][2])
            y = random.randint(box[0][1],box[0][3])
            if y == 640:
                y = 639
            if x == 640:
                x = 639'''
        self.position = [x,y]
        hot = random.uniform(0,1)
        if hot > 0.5:
            self.value = 0.9 
        else:
            self.value = 0.1
        
        self.length = L * (box[3]  - box[1])
        self.angle = random.uniform(0,np.pi)
        self.Infrared_block = [self.position,self.value,self.length,self.angle]
        

    def set_para(self,position,length,angle):
        self.position = position
        self.length = length
        self.angle = angle
        Infrared_block = [self.position,self.value,self.length,self.angle]
        return Infrared_block
    
    def return_para(self):
        return self.Infrared_block
        
        
def applied_blocks(block_mask,block_image, block, value):
    
    position = block[0]
    x = position[0]
    y = position[1]
    
    value = block[1]
    length = block[2]
    angle = block[3]
   
    '''if value > 0.5:
        width = length * 0.74
    elif value < 0.5:'''
    width = length * 0.45
    
    
    corner1 = [np.array(round(x)),np.array(round(y))]
    
    corner2 = [np.array(round(length * np.sin(angle) + x)), np.array(round(length * np.cos(angle) + y))]
    corner4 = [np.array(round(width * np.cos(angle) + x)), np.array(round(-width * np.sin(angle) + y))]
    corner3 = [np.array(round(width * np.cos(angle) + corner2[0])), np.array(round(-width * np.sin(angle) + corner2[1]))]
    corners = np.array([[corner1,corner2,corner3,corner4]],dtype=np.int32)
    
    block_mask = block_mask.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8).copy()
    block_image = block_image.squeeze(0).cpu().permute(1,2,0).numpy().astype(np.uint8).copy()

    block_mask = cv2.fillPoly(block_mask,corners,1)     
    block_image = cv2.fillPoly(block_image,corners,(value,value,value))
    block_mask = torch.from_numpy(block_mask).unsqueeze(0).permute(0,3,1,2)
    block_image = torch.from_numpy(block_image).unsqueeze(0).permute(0,3,1,2)
    #block_mask = kornia.utils.draw.draw_convex_polygon(block_mask,corners,colors)     
    #block_image = kornia.utils.draw.draw_convex_polygon(block_image,corners,value)
    
    return block_mask, block_image



def random_block(box,img_batch,model,config,threshold):
    x1,y1,x2,y2 = box[0],box[1],box[2],box[3]
    cold_block = torch.ones_like(img_batch)
    blocks = []
    length = 8
    best_score = 100
    best_image = img_batch
    mask = torch.zeros_like(img_batch[:,0,:,:]).unsqueeze(0)
    
    for num1 in range(1000):
        Blocks = []
        blocked_image = img_batch
        for num in range(config.best_num_block):
            Infrared_block = InfraredBlock(L=config.length,box=box)
            Blocks.append(Infrared_block.return_para())
            block = Blocks[num]
            block_mask, block_image = applied_blocks(mask,img_batch,block,block[1])
            block_mask = block_mask.to(device)
            block_image = block_image.to(device)
            blocked_image = blocked_image * (1 - block_mask) + block_image * block_mask

        #img_batch_applied = img_batch * (1-mask) + cold_block * mask
        
        
        
        #rects_attacked = self.thermal_detector(image__applied_gray,upsample_num_times=1)
        #[rects_attacked, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple([self.thermal_detector], image__applied_gray, upsample_num_times=1, adjust_threshold=0.0)
        max_score = computation_score(model,config,blocked_image)
        if num1 % 50 ==0:
            print('best_score=',max_score)
        if max_score < best_score:
            best_score = max_score
            best_image = blocked_image
        if best_score < threshold:
            return best_score
        save_img2 = best_image.squeeze(0).permute(1,2,0).cpu().numpy()
       
        cv2.imwrite('processing_images/111thermal_image.png',save_img2*255)
    return best_score
        
    if best_score < 0.3:
        success_case_3+=1
    if best_score < 0.4 :
        success_case_4+=1
    if best_score < 0.5 :
        success_case_5+=1
    if best_score < 0.6:
        success_case_6+=1
    if best_score < 0.7:
        success_case_7+=1
    
    print('max_score=',best_score)
    total_case=1





def evo_diff_block(box,img_batch,model,config,threshold):
    x1,y1,x2,y2 = box[0],box[1],box[2],box[3]
    cold_block = torch.ones_like(img_batch)
    blocks = []
    length = 8
    best_score = 100
    best_image = img_batch
    mask = torch.zeros_like(img_batch[:,0,:,:])
    
    Blocks = []
    for j in range(config.num_block):
        Infrared_block = InfraredBlock(L=config.length,box=box)
        Blocks.append(Infrared_block.return_para())
    
    
    Blocks,best_score = diff_evo(Blocks,img_batch,config.best_num_block,box,model,config,threshold)
    return best_score    
        
        
    
    
    
    
def diff_evo(Blocks,image,number_best,box,model,config,threshold):
    success = 0
    best_scores = 1
    best_score_diff = 0
    all_blocks = []
    step = 1000
    D=4    #单条染色体基因数目
    CR=0.1  #交叉算子
    F0=0.4  #初始变异算子
    
    G = 100
    Rm = 0.5
    Rc = 0.6
    best_blocks = []
    for num in range(number_best):
        best_blocks.append(Blocks[num])
    for i in range(step):
        
        selected_block = []
        
        NP = len(Blocks) # 3 个
        #print('all_blocks_shape=',np.array(all_blocks).shape)
        
        #变异操作，和遗传算法的变异不同！,得到任意两个个体的差值，与变异算子相乘加第三个个体
        
        for m in range(number_best):
            block_mask = torch.zeros_like(image)[:,0,:,:].unsqueeze(0)
            block_image = torch.zeros_like(image)

            r1=m
            r2=random.sample(range(0,NP),1)[0]
            r3=random.sample(range(0,NP),1)[0]
            
            while Blocks[r2][0][0] == best_blocks[r1][0] and Blocks[r2][0][1] == best_blocks[r1][1]:
                r2=random.sample(range(0,NP),1)[0]
            
            while Blocks[r3][0][0] == best_blocks[r1][0] and Blocks[r3][0][1] == best_blocks[r1][1] or \
                Blocks[r3][0][0] == Blocks[r2][0][0] and Blocks[r3][0][1] == Blocks[r2][0][1]:
                r3=random.sample(range(0,NP),1)[0]
    
            
            Parents = best_blocks[r1]#Blocks[r1]
            
            Blocks[r1][0][0] = Blocks[r1][0][0]+Rm*(Blocks[r2][0][0]-Blocks[r3][0][0])
            Blocks[r1][0][1] = Blocks[r1][0][1]+Rm*(Blocks[r2][0][1]-Blocks[r3][0][1])
            Blocks[r1][3] = Blocks[r1][3]+Rm*(Blocks[r2][3]-Blocks[r3][3])

            if Blocks[r1][0][0] > box[2] or Blocks[r1][0][0] < box[0]:
                Blocks[r1][0][0] = random.randint(box[0],box[2])
            if Blocks[r1][0][1] > box[3] or Blocks[r1][0][1] < box[1]:
                Blocks[r1][0][1] = random.randint(box[1],box[3])
            if Blocks[r1][3] > np.pi or Blocks[r1][3] < 0:
                Blocks[r1][3] = random.uniform(0,np.pi)

                    
              
            
            Children = Blocks[r1]
            
            for d in range(D):
                rand = random.random()
                if rand <= Rc:
                    Children[d] = Parents[d]
                    
            
            selected_block = best_blocks
            selected_block[r1] = Children
            
            for x in range(len(selected_block)):
                using_block = selected_block[x]
                #using_block = aug_block(using_block)
                block_mask,block_image = applied_blocks(block_mask,block_image,using_block,using_block[1])
            block_mask = block_mask.to(device)
            block_image = block_image.to(device)
            blocked_image = image * (1 - block_mask) + block_image * block_mask
            
            max_score = computation_score(model,config,blocked_image)
            
            #print('len3=',len(Blocks))
            if max_score < best_scores:
                best_blocks[r1] = Children
                best_scores = max_score
                best_image = blocked_image
            
           
            save_img2 = best_image.squeeze(0).permute(1,2,0).cpu().numpy()
            if i % 50 ==0:
                print('best_score=',best_scores)
            cv2.resize(save_img2,(416,416))
            cv2.imwrite('processing_images/maskthermal_image.png',save_img2*255)
            if best_scores<threshold:
                return best_blocks,best_scores
            
    
    return best_blocks,best_scores


def aug_block(using_block):
    random_number1 = random.uniform(-1, 1)
    using_block[0][0] = using_block[0][0] + random_number1*2
    random_number2 = random.uniform(-1, 1)
    using_block[0][1] = using_block[0][1] + random_number2*2
    random_number3 = random.uniform(-1, 1)
    using_block[1] = using_block[1] + random_number3*0.05
    '''random_number4 = random.uniform(-1, 1)
    using_block[2] = using_block[2] + random_number4*1'''
    random_number5 = random.uniform(-1, 1)
    using_block[3] = using_block[3] + random_number5*0.01

    return using_block
    
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

def computation_score(model,config,img_batch):
    if config.detector == 'yolov5':
        save_img_sub = img_batch.squeeze(0).cpu().detach().numpy().transpose(1,2,0)
        cv2.imwrite('processing_images/thermalimg_batch123123.jpg',save_img_sub[:,:,::-1]*255)
        out, train_out = model(img_batch)
        obj_confidence = out[:, :, 4]
        max_obj_confidence, _ = torch.max(obj_confidence, dim=1)
        obj_loss = torch.mean(max_obj_confidence)
    elif config.detector == 'yolov8':   
        sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
        sub_img_batch_applied=sub_img_batch_applied.astype(np.uint8)      
        score= model.detect(sub_img_batch_applied) 
        
        score = torch.from_numpy(np.array(score))
        #score = torch.sigmoid(score)
        max_score = torch.max(score)
        obj_loss = torch.mean(max_score) 
    elif config.detector == 'retinaface':
        #sub_img_batch_applied = p_img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)
        
        det_loss = model.predict_jsons(img_batch)
        obj_loss = det_loss.to(device)
    elif config.detector == 'dlib':
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
    elif config.detector =='opencv_dnn':
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
    elif config.detector =='scrfd':
        sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
        sub_img_batch_applied=sub_img_batch_applied.astype(np.uint8)
        score = model.get(sub_img_batch_applied)
        if len(score)!=0:
            obj_loss = torch.max(torch.from_numpy(np.array(score))).to(device)
        else:
            obj_loss = 0
    elif config.detector =='mtcnn':
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
        

    elif config.detector =="dsfd":
        target_size = (128, 128)
        sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
        sub_img_batch_applied = sub_img_batch_applied.astype(np.uint8)
        detections = model.detect_on_image(sub_img_batch_applied, target_size, device, is_pad=False, keep_thresh=0.1)
        score = detections[:,-1]
        if len(score)!=0:
            obj_loss = torch.max(torch.from_numpy(np.array(score))).to(device)
        else:
            obj_loss = 0
    elif config.detector =="ulfd":
        sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
        sub_img_batch_applied = sub_img_batch_applied.astype(np.uint8)
        result = model(sub_img_batch_applied)
        scores = result['scores']
        if len(scores)!=0:
            obj_loss = torch.max(torch.from_numpy(np.array(scores))).to(device)
        else:
            obj_loss = 0
    elif config.detector =="mogface":
        p_img_batch = F.interpolate(img_batch,(256,256))
        result = model(p_img_batch)
        scores = torch.max(result[0].squeeze(0))
        obj_loss = scores
    return obj_loss


class AdversarialMask:
    def __init__(self, config):
        self.config = config
        set_random_seed(seed_value=self.config.seed)

       

        face_landmark_detector = function.get_landmark_detector(self.config, device)
        self.location_extractor = LandmarkExtractor(device, face_landmark_detector, self.config.img_size).to(device)
        self.fxz_projector = FaceXZooProjector(device, self.config.img_size, self.config.patch_size,config).to(device)
        self.total_variation = TotalVariation(device).to(device)
       
        self.nps_calculator = NPSCalculator('non_printability/30values.txt', config.patch_size[0],device=device)

        self.best_patch = None
        self.pre_model = load_model('weights/yolov5n_face.pt', device)
        #self.model = YoloDetector(target_size=720, device=device, min_face=90)
        self.thermal_detector,self.thermal_predictor = function.get_thermal_predictor(device)
        self.Test_dataset = image_Testdata('datasets/thermal/gray/train/images',self.config)#datasets/thermal/gray/train/images
        #self.Test_dataset = image_Testdata('datasets/thermal_images_physical',self.config)
        #self.Test_dataset = torch.utils.data.DataLoader(Test_dataset,batch_size = 8,shuffle=False)
        
        if self.config.detector =='yolov5':
            self.model = load_model('weights/yolov5n_face.pt', device)
        elif self.config.detector =='yolov8':
            self.model =  YOLOv8_face('weights/yolov8n-face.onnx', conf_thres=self.config.threshold, iou_thres=0.4)
        elif  self.config.detector =='retinaface':
            self.model = get_model("resnet50_2020-07-20", max_size=416,device=device)
        elif  self.config.detector =='dlib':
            self.model = dlib.cnn_face_detection_model_v1('models/mmod_human_face_detector.dat')
        elif self.config.detector =='opencv_dnn':
            self.model = cv2.dnn.readNetFromTensorflow('models/opencv_face_detector_uint8.pb','models/opencv_face_detector.pbtxt')
        elif  self.config.detector =='scrfd':
            self.model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.model.prepare(ctx_id=0, det_size=(640, 640))
        elif  self.config.detector =='mtcnn':
            self.model  = FaceDetector()
        elif self.config.detector =="dsfd":
            self.model = SSD("test")
            self.model.load_state_dict(torch.load('models/DSFD/weights/WIDERFace_DSFD_RES152.pth'))
            self.model.to(device).eval()
        elif self.config.detector =="ulfd":
            self.model = pipeline(Tasks.face_detection, 'damo/cv_manual_face-detection_ulfd')
        elif self.config.detector =="mogface":
            #self.model = pipeline(Tasks.face_detection, 'damo/cv_resnet101_face-detection_cvpr22papermogface')
            config = 'models/MogFace/configs/mogface/MogFace_E.yml'
            # generate det_info and det_result
            cfg = load_config(config)
            cfg['phase'] = 'test'
            if 'use_hcam' in cfg and cfg['use_hcam']:
                # test_th
                cfg['fp_th'] = 0.12
            self.model = create(cfg.architecture)
            self.model.load_state_dict(torch.load('models/MogFace/configs/mogface/model_140000.pth'))
            self.model.to(device)
   

    def test(self):
        adv_patch_cpu = function.load_mask(self.config,'data/masks/black.png',device)
        adv_patch_cpu[:,:3,:,:] = torch.zeros_like(adv_patch_cpu[:,:3,:,:]) + 0.1
        success = 1
        total = 0
        success_case_3 =0
        success_case_4 = 0
        success_case_5 = 0
        success_case_6 = 0
        success_case_7 = 0
        for i, img_batch in enumerate(self.Test_dataset):
            img_batch = img_batch.to(device).float()
            success_num,total_num = self.forward_step(img_batch, adv_patch_cpu,i,0.6)
           
            success_case_3 += success_num[0]
            success_case_4 += success_num[1]
            success_case_5 += success_num[2]
            success_case_6 += success_num[3]
            success_case_7 += success_num[4]
            total += total_num
            if total == 0:
                total = 1
            success_rate_3 = success_case_3/total
            success_rate_4 = success_case_4/total
            success_rate_5 = success_case_5/total
            success_rate_6 = success_case_6/total
            success_rate_7 = success_case_7/total
            
            print('i=%d,success rate 0.3=%f,success rate 0.4=%f,success rate 0.5=%f,success rate 0.6=%f,success rate 0.7=%f,best_num=%d,length=%f,optim_cold=%s,model=,%s'%(i,success_rate_3,success_rate_4,success_rate_5,success_rate_6,success_rate_7,self.config.best_num_block,self.config.length,self.config.infrared_method,self.config.detector))
            
   

    def forward_step(self, img_batch, adv_patch_cpu,ind,threshold):
        success_case_3 = 0
        success_case_4 = 0
        success_case_5 = 0
        success_case_6 = 0
        success_case_7 = 0
        total_case = 0
        fail_case = 0
        adv_patch = adv_patch_cpu.to(device)
        #image_gray = cv2.cvtColor(img_batch, cv2.COLOR_BGRA2GRAY)
        
        
        #save_img1 = img_batch.copy()
        if self.config.mode == 'train':
            if self.config.detector == 'yolov5':
                
                out, train_out = self.model(img_batch)
                obj_confidence = out[:, :, 4]
                max_obj_confidence, _ = torch.max(obj_confidence, dim=1)
                obj_loss = torch.mean(max_obj_confidence)
        
            elif self.config.detector == 'yolov8':   
                transform = transforms.Compose([transforms.Resize((128,128)),
                                               transforms.ToTensor()])
                
                ###thermal
                #img_path = 'datasets/image4.jpg'
                '''image = Image.open(img_path)#.convert('RGB')
                image = transform(image).unsqueeze(0).to(device)'''
                sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
                sub_img_batch_applied=sub_img_batch_applied.astype(np.uint8)      
                score= self.model.detect(sub_img_batch_applied) 
                score = torch.from_numpy(np.array(score))
                #score = torch.sigmoid(score)
                max_score = torch.max(score)
                obj_loss = torch.mean(max_score) 
            elif self.config.detector == 'retinaface':
                #sub_img_batch_applied = p_img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)
                
                det_loss = self.model.predict_jsons(img_batch)
                obj_loss = det_loss.to(device)
            elif self.config.detector == 'dlib':
                sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
                sub_img_batch_applied=sub_img_batch_applied.astype(np.uint8)
                sub_img_batch_applied = imutils.resize(sub_img_batch_applied, width=300)
                # convert the image to grayscale
                sub_img_batch_applied = cv2.cvtColor(sub_img_batch_applied, cv2.COLOR_BGRA2GRAY)
                # detect faces in the image 
                rects = self.model(sub_img_batch_applied, upsample_num_times=1)    
                if len(rects) !=0:
                    rect = rects[0]
                    confidence = rect.confidence
                    det_loss = torch.tensor(confidence)
                    obj_loss = det_loss.to(device)
                else:
                    obj_loss =  0
            elif self.config.detector =='opencv_dnn':
                sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
                sub_img_batch_applied=sub_img_batch_applied.astype(np.uint8)
                blob = cv2.dnn.blobFromImage(sub_img_batch_applied,1.0,(112,112),[104,117,123],False,False)
                frameHeight = sub_img_batch_applied.shape[0]
                frameWidth = sub_img_batch_applied.shape[1]
                self.model.setInput(blob)
                detections = self.model.forward()
                bboxes = []
                ret = 0 
                max_score = torch.from_numpy(detections[:,:,:,2])
                #max_score = torch.sigmoid(max_score)
                obj_loss = torch.max(max_score).to(device)
            elif self.config.detector =='scrfd':
                sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
                sub_img_batch_applied=sub_img_batch_applied.astype(np.uint8)
                score = self.model.get(sub_img_batch_applied)
                if len(score)!=0:
                    obj_loss = torch.max(torch.from_numpy(np.array(score))).to(device)
                else:
                    obj_loss = 0
            elif self.config.detector =='mtcnn':
                sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
                sub_img_batch_applied = sub_img_batch_applied.astype(np.uint8)
                sub_img_batch_applied = Image.fromarray(sub_img_batch_applied)
                bboxes, landmarks = self.model.detect(sub_img_batch_applied)
                
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
                

            elif self.config.detector =="dsfd":
                target_size = (128, 128)
                sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
                sub_img_batch_applied = sub_img_batch_applied.astype(np.uint8)
                detections = self.model.detect_on_image(sub_img_batch_applied, target_size, device, is_pad=False, keep_thresh=0.1)
                score = detections[:,-1]
                if len(score)!=0:
                    obj_loss = torch.max(torch.from_numpy(np.array(score))).to(device)
                else:
                    obj_loss = 0
            elif self.config.detector =="ulfd":
                sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
                sub_img_batch_applied = sub_img_batch_applied.astype(np.uint8)
                result = self.model(sub_img_batch_applied)
                scores = result['scores']
                if len(scores)!=0:
                    obj_loss = torch.max(torch.from_numpy(np.array(scores))).to(device)
                else:
                    obj_loss = 0
            elif self.config.detector =="mogface":
                p_img_batch = F.interpolate(img_batch,(256,256))
                result = self.model(p_img_batch)
                scores = torch.max(result[0].squeeze(0))
                obj_loss = scores
            print('obj_loss=',obj_loss)
            if obj_loss < 0.0:
                return [0,0,0,0,0],0
        




            preds = self.pre_model(img_batch)[0]
            
        # Apply NMS
            pred = non_max_suppression_face(preds)
        
            if len(pred[0])==0:
                return [0,0,0,0,0],0
            
            x_min = pred[0][0][0]
            y_min = pred[0][0][1]
            x_max = pred[0][0][2]
            y_max = pred[0][0][3]
            width = x_max - x_min
            height = y_max - y_min
            x1 = int(x_min)
            y1 = int(y_min + height/5)
            x2 = int(x_max)
            y2 = int(y_max - height/2)
            
            
            box = [x1,y1,x2,y2]
            ###rectangle mask
            if self.config.infrared_method == 'random':
                #best_score = computation_score(self.model,self.config,img_batch)
                best_score = random_block(box,img_batch,self.model,self.config,threshold)
                if best_score < 0.3:
                    success_case_3+=1
                if best_score < 0.4 :
                    success_case_4+=1
                if best_score < 0.5 :
                    success_case_5+=1
                if best_score < 0.6:
                    success_case_6+=1
                if best_score < 0.7:
                    success_case_7+=1

                total_case=1
                print('best_score=',best_score)
            elif self.config.infrared_method == 'evo_diff':
                best_score = evo_diff_block(box,img_batch,self.model,self.config,threshold)
                
                if best_score < 0.3:
                    success_case_3+=1
                if best_score < 0.4 :
                    success_case_4+=1
                if best_score < 0.5 :
                    success_case_5+=1
                if best_score < 0.6:
                    success_case_6+=1
                if best_score < 0.7:
                    success_case_7+=1
                    
                print('max_score=',best_score)
                total_case=1
            #####load mask
            elif self.config.infrared_method == 'mask_block':
                
                image_numpy = img_batch.squeeze(0).permute(1,2,0).cpu().numpy()*255
                image_gray = cv2.cvtColor(image_numpy, cv2.COLOR_BGRA2GRAY).astype('uint8')
                rects = self.thermal_detector(image_gray,upsample_num_times=1)
                
                if len(rects)==0:
                    return [0,0,0,0,0], 0
                preds = []
                for rect in rects:
                # convert the dlib rectangle into an OpenCV bounding box and
                # draw a bounding box surrounding the face
                    (x, y, w, h) = face_utils.rect_to_bb(rect)
                    #cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # predict the location of facial landmark coordinates, 
                    # then convert the prediction to an easily parsable NumPy array
                    shape = self.thermal_predictor(image_gray, rect)
                    pred = face_utils.shape_to_np(shape)
                    
                    
                    
                preds.append(pred)
                preds = torch.from_numpy(np.array(preds)).to(device)
                
                if adv_patch_cpu.shape[1]==4:
                    img_batch_applied,mask1 = self.fxz_projector(img_batch, preds, adv_patch_cpu[:,:3], uv_mask_src=adv_patch_cpu[:,3], is_3d=True,do_aug=False,first=False)
                else:
                ####random generate mask
                    img_batch_applied,mask1 = self.fxz_projector(img_batch, preds, adv_patch, do_aug=False)
                #img_batch_applied = img_batch_applied.squeeze(0).cpu().detach().numpy().transpose(1,2,0)
                
                #img_batch_applied = F.interpolate(img_batch_applied,(400,300))
                best_score = evo_diff_block(box,img_batch_applied,self.model,self.config,threshold)
                
                if best_score < 0.3:
                    success_case_3+=1
                if best_score < 0.4 :
                    success_case_4+=1
                if best_score < 0.5 :
                    success_case_5+=1
                if best_score < 0.6:
                    success_case_6+=1
                if best_score < 0.7:
                    success_case_7+=1
                    
                print('max_score=',best_score)
                total_case=1
        elif self.config.mode == 'test':
            if self.config.detector == 'yolov5':
                out, train_out = self.model(img_batch)
                obj_confidence = out[:, :, 4]
                max_obj_confidence, _ = torch.max(obj_confidence, dim=1)
                obj_loss = torch.mean(max_obj_confidence)
            elif self.config.detector == 'yolov8':   
                sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
                sub_img_batch_applied=sub_img_batch_applied.astype(np.uint8)      
                score= self.model.detect(sub_img_batch_applied) 
                score = torch.from_numpy(np.array(score))
                #score = torch.sigmoid(score)
                max_score = torch.max(score)
                obj_loss = torch.mean(max_score) 
            elif self.config.detector == 'retinaface':
                #sub_img_batch_applied = p_img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)
                
                det_loss = self.model.predict_jsons(img_batch)
                obj_loss = det_loss.to(device)
            elif self.config.detector == 'dlib':
                sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
                sub_img_batch_applied=sub_img_batch_applied.astype(np.uint8)
                sub_img_batch_applied = imutils.resize(sub_img_batch_applied, width=300)
                # convert the image to grayscale
                sub_img_batch_applied = cv2.cvtColor(sub_img_batch_applied, cv2.COLOR_BGRA2GRAY)
                # detect faces in the image 
                rects = self.model(sub_img_batch_applied, upsample_num_times=1)    
                if len(rects) !=0:
                    rect = rects[0]
                    confidence = rect.confidence
                    det_loss = torch.tensor(confidence)
                    obj_loss = det_loss.to(device)
                else:
                    obj_loss =  0
            elif self.config.detector =='opencv_dnn':
                sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
                sub_img_batch_applied=sub_img_batch_applied.astype(np.uint8)
                blob = cv2.dnn.blobFromImage(sub_img_batch_applied,1.0,(112,112),[104,117,123],False,False)
                frameHeight = sub_img_batch_applied.shape[0]
                frameWidth = sub_img_batch_applied.shape[1]
                self.model.setInput(blob)
                detections = self.model.forward()
                bboxes = []
                ret = 0 
                max_score = torch.from_numpy(detections[:,:,:,2])
                #max_score = torch.sigmoid(max_score)
                obj_loss = torch.max(max_score).to(device)
            elif self.config.detector =='scrfd':
                sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
                sub_img_batch_applied=sub_img_batch_applied.astype(np.uint8)
                score = self.model.get(sub_img_batch_applied)
                if len(score)!=0:
                    obj_loss = torch.max(torch.from_numpy(np.array(score))).to(device)
                else:
                    obj_loss = 0
            elif self.config.detector =='mtcnn':
                sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
                sub_img_batch_applied = sub_img_batch_applied.astype(np.uint8)
                sub_img_batch_applied = Image.fromarray(sub_img_batch_applied)
                bboxes, landmarks = self.model.detect(sub_img_batch_applied)
                
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
                

            elif self.config.detector =="dsfd":
                target_size = (128, 128)
                sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
                sub_img_batch_applied = sub_img_batch_applied.astype(np.uint8)
                detections = self.model.detect_on_image(sub_img_batch_applied, target_size, device, is_pad=False, keep_thresh=0.1)
                score = detections[:,-1]
                if len(score)!=0:
                    obj_loss = torch.max(torch.from_numpy(np.array(score))).to(device)
                else:
                    obj_loss = 0
            elif self.config.detector =="ulfd":
                sub_img_batch_applied = img_batch.cpu().detach().squeeze(0).numpy().transpose(1,2,0)*255
                sub_img_batch_applied = sub_img_batch_applied.astype(np.uint8)
                result = self.model(sub_img_batch_applied)
                scores = result['scores']
                if len(scores)!=0:
                    obj_loss = torch.max(torch.from_numpy(np.array(scores))).to(device)
                else:
                    obj_loss = 0
            elif self.config.detector =="mogface":
                p_img_batch = F.interpolate(img_batch,(256,256))
                result = self.model(p_img_batch)
                scores = torch.max(result[0].squeeze(0))
                obj_loss = scores
            if obj_loss < 0.3:
                success_case_3+=1
            if obj_loss < 0.4 :
                success_case_4+=1
            if obj_loss < 0.5 :
                success_case_5+=1
            if obj_loss < 0.6:
                success_case_6+=1
            if obj_loss < 0.7:
                success_case_7+=1
            print('obj_loss=',obj_loss)
        success_case = [success_case_3,success_case_4,success_case_5,success_case_6,success_case_7]
        total_case = 1
        return success_case, total_case
    
    '''
    def ev_algorithm(self,Blocks, x_corner,length,image_gray,img_batch,cold_block):
        D=4    #单条染色体基因数目
        CR=0.1  #交叉算子
        F0=0.4  #初始变异算子
        step = 1000  #最大遗传代数
        G = 100
        Rm = 0.5
        Rc = 0.6
        x1 = x_corner[0]
        y1 = x_corner[1]
        x2 = x_corner[2]
        y2 = x_corner[3]
        NP = len(Blocks)
        best_score = 100
        best_image = image_gray
        
        for i in range(step):
            best_blocks = []
            mask = np.zeros_like(img_batch)
            for m in range(NP):
                r1=m
                r2=random.sample(range(0,NP),1)[0]
                while r2==m or r2==r1:
                    r2=random.sample(range(0,NP),1)[0]
                r3=random.sample(range(0,NP),1)[0]
                while r3==m or r3==r2 or r3==r1:
                    r3=random.sample(range(0,NP),1)[0]
                
                Parents = Blocks[r1]
                
                Blocks[r1][0] = Blocks[r1][0]+Rm*(Blocks[r2][0]-Blocks[r3][0])
                Blocks[r1][1] = Blocks[r1][1]+Rm*(Blocks[r2][1]-Blocks[r3][1])
                
        
                if Blocks[r1][0]<x1 or Blocks[r1][0]>x2:
                    Blocks[r1][0] = random.randint(x1,x2-length-1)
                if Blocks[r1][1]<y1 or Blocks[r1][1]>y2:
                    Blocks[r1][1] = random.randint(y1,y2-length-1)
                    
                    
                for num in range(5):
                    block_x = int(Blocks[r1][0])
                    block_y = int(Blocks[r1][1])
                    mask[block_y:block_y+length,block_x:block_x+length,:] = 1
            
            
                img_batch_applied = img_batch * (1-mask) + cold_block * mask
                
                img_batch_applied = img_batch_applied.astype(np.float32)
                image_applied_gray = cv2.cvtColor(img_batch_applied, cv2.COLOR_BGRA2GRAY).astype(np.uint8)
                
    
                [rects_attacked, confidences, detector_idxs] = dlib.fhog_object_detector.run_multiple([self.thermal_detector], image_applied_gray, upsample_num_times=1, adjust_threshold=0.0)
                if len(confidences)!=0:
                    max_score = np.max(confidences)
                else: 
                    max_score = 0
                
                if max_score < best_score:
                    best_score = max_score
                    best_image = image_applied_gray
                
                save_img2 = best_image
                save_img2 = best_image
                cv2.imwrite('processing_images/222thermal_image.jpg',save_img2)
                print('best_score=',best_score)
        return best_score, best_image
'''
def main():
    mode = 'universal'
    config = patch_config_types[mode]()
    print('Starting test...', flush=True)
    adv_mask = AdversarialMask(config)
    with torch.no_grad():
        adv_mask.test()
    print('Finished test...', flush=True)


if __name__ == '__main__':
    main()

