from PIL import Image, ImageDraw
from torchvision import datasets, transforms
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import scipy
import torch
import torchvision.models as models
import os
import cv2
from models import *
import warnings
import imutils
import shutil
from face_detector import YoloDetector, YOLOv8_face
from retinaface.pre_trained_models import get_model
#from retinaface.predict_single import model
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from models.mtcnn import FaceDetector
from models.DSFD.face_ssd_infer import SSD
import dlib
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from models.MogFace.core.workspace import register, create, global_config, load_config
import argparse
from dataloader import *
from utils import rotate
from utils import stick
from utils import mapping3d
from utils import feature
warnings.filterwarnings("ignore")
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
""" perturb the image """
def perturb_image(xs, backimg, sticker,opstickercv,magnification, zstore, searchspace, facemask):
    xs = np.array(xs)
   
    #print('imgs=',image_arr.shape)
    d = xs.ndim
    if(d==1):
        xs = np.array([xs])
    w,h = backimg.size
    
    imgs = []
    valid = []
    l = len(xs)
    for i in range(l):
        #print('making {}-th perturbed image'.format(i),end='\r')
        sid = int(xs[i][0])
        x = int(searchspace[sid][0])
        y = int(searchspace[sid][1])
        angle = xs[i][2]
        rt_sticker = rotate.rotate_bound_white_bg(opstickercv, angle)
        
        nsticker,_ = mapping3d.deformation3d(sticker,rt_sticker,magnification,zstore,x,y)
        
        outImage = stick.make_stick2(backimg=backimg, sticker=nsticker, x=x, y=y, factor=xs[i][1])
        
        imgs.append(outImage)
        
        check_result = int(check_valid(w, h, nsticker, x, y, facemask))
        valid.append(check_result)
        #print('outImage=',np.array(outImage).shape)
        #cv2.imwrite('output.jpg',np.array(outImage)[:,:,::-1])
    return imgs,valid

def check_valid(w, h, sticker, x, y, facemask):
    _,basemap = stick.make_basemap(width=w, height=h, sticker=sticker, x=x, y=y)
    area = np.sum(basemap)
    overlap = facemask * basemap
    retain = np.sum(overlap)
    if(abs(area - retain) > 15):
        return 0
    else:
        return 1

def simple_perturb(xs, backimg, sticker, searchspace, facemask):
    xs = np.array(xs)
    d = xs.ndim
    if(d==1):
        xs = np.array([xs])
    w,h = backimg.size
    
    imgs = []
    valid = []
    l = len(xs)
    for i in range(l):
        print('making {}-th perturbed image'.format(i),end='\r')
        sid = int(xs[i][0])
        x = int(searchspace[sid][0])
        y = int(searchspace[sid][1])
        angle = xs[i][2]
        stickercv = rotate.img_to_cv(sticker)
        rt_sticker = rotate.rotate_bound_white_bg(stickercv, angle)
        outImage = stick.make_stick2(backimg=backimg, sticker=rt_sticker, x=x, y=y, factor=xs[i][1])
        imgs.append(outImage)
        
        check_result = int(check_valid(w, h, rt_sticker, x, y, facemask))
        valid.append(check_result)
    
    return imgs,valid

""" query the model for image's classification """
"""
def predict_type_xxx(image_perturbed, cleancrop):
    typess = [[top-1,...,top-5],...]
    percent = [[probability vector],...]
    return typess,percent
"""
def predict_type_facenet(image_perturbed, cleancrop):
    #print('shape = ',image_perturbed.shape)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #print('Running on device: {}'.format(device))
    def collate_fn(x):
        return x
    loader = DataLoader(
        image_perturbed,
        batch_size=42,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    #resnet = torch.load('./models/facenet/net_13_022.pth',map_location='cuda:0').to(device)
    #resnet.eval()
    resnet = InceptionResnetV1(pretrained='casia-webface').to(device).eval()
    resnet.classify = True
    
    percent = []
    typess = []
    
    for X in loader:
        C = mtcnn(X)   # return tensor list
        C = [cleancrop if x is None else x for x in C]
        batch_t = torch.stack(C)
        #print(batch_t.shape)
        batch_t = batch_t.to(device)
        out = resnet(batch_t).cpu()
        
        #print('logits\' len = ',len(out[0]))
        #print('true label = ',true_label)
        with torch.no_grad():
            _, indices = torch.sort(out.detach(), descending=True)
            percentage = torch.nn.functional.softmax(out.detach(), dim=1) * 100
            
            for i in range(len(out)):
                cla = [indices[i][0].item(),indices[i][1].item(),indices[i][2].item(),\
                       indices[i][3].item(),indices[i][4].item()]
                typess.append(cla)
                tage = percentage[i]
                percent.append(tage)
    
    return typess,percent

"""
def initial_predict_xxx(image_perturbed):
    typess = [[top-1,...,top-5],...]
    percent = [[probability vector],...]
    C = mtcnn(image_perturbed,save_path='./test.jpg')
    return typess,percent,C[0]
"""
def initial_predict_facenet(image_perturbed):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    #resnet = torch.load('./models/facenet_pytorch/net_13_022.pth',map_location='cuda:0').to(device)
    #resnet.eval()
    resnet = InceptionResnetV1(pretrained='casia-webface').to(device).eval()
    resnet.classify = True
    
    percent = []
    typess = []
    
    C = mtcnn(image_perturbed,save_path='./test.jpg')   # return tensor list
    batch_t = torch.stack(C)
    #print(batch_t.shape)
    batch_t = batch_t.to(device)
    out = resnet(batch_t).cpu()
    with torch.no_grad():
        _, indices = torch.sort(out.detach(), descending=True)
        percentage = torch.nn.functional.softmax(out.detach(), dim=1) * 100
        cla = [indices[0][0].item(),indices[0][1].item(),indices[0][2].item(),\
               indices[0][3].item(),indices[0][4].item()]
        typess.append(cla)
        tage = percentage[0]
        percent.append(tage)

    return typess,percent,C[0]



def initial_face_detector(detector):
    
    
    if detector =='yolov5':
        model = YoloDetector(target_size=720, device=device, min_face=90)
    elif detector =='yolov8':
        model =  YOLOv8_face('weights/yolov8n-face.onnx', conf_thres=0.1, iou_thres=0.4)
    elif  detector =='retinaface':
        model = get_model("resnet50_2020-07-20", max_size=416,device=device)
        
    elif  detector =='dlib':
        model = dlib.cnn_face_detection_model_v1('models/mmod_human_face_detector.dat')
    elif detector =='opencv_dnn':
        model = cv2.dnn.readNetFromTensorflow('models/opencv_face_detector_uint8.pb','models/opencv_face_detector.pbtxt')
    elif  detector =='scrfd':
        model = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        model.prepare(ctx_id=0, det_size=(128, 128))
    elif  detector =='mtcnn':
        model  = FaceDetector()
    elif detector =="dsfd":
        model = SSD("test")
        model.load_state_dict(torch.load('models/DSFD/weights/WIDERFace_DSFD_RES152.pth'))
        model.to(device).eval()
    elif detector =="ulfd":
        model = pipeline(Tasks.face_detection, 'damo/cv_manual_face-detection_ulfd')
    elif detector =="mogface":
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
        '''config = 'models/mogface/MogFace_E.yml'
        
        cfg = load_config(config)
        
        cfg['phase'] = 'test'
        resnet = ResNet(depth= 152)
        lfpn = LFPN(c2_out_ch= 256,
                    c3_out_ch= 512,
                    c4_out_ch= 1024,
                    c5_out_ch= 2048,
                    c6_mid_ch= 512,
                    c6_out_ch= 512,
                    c7_mid_ch= 128,
                    c7_out_ch= 256,
                    out_dsfd_ft= True)
        mogface = MogPredNet(num_anchor_per_pixel= 1,
                            input_ch_list= [256, 256, 256, 256, 256, 256],
                            use_deep_head= True,
                            deep_head_with_gn= True ,
                            deep_head_ch= 256,
                            use_ssh= False,
                            use_cpm= True,
                            use_dsfd= True,
                            retina_cls_weight_init_fn= 'RetinaClsWeightInit')
        model = WiderFaceBaseNet(backbone=resnet,fpn=lfpn,pred_net=mogface)
        model.load_state_dict(torch.load('models/mogface/model_140000.pth'))'''
        #model = pipeline(Tasks.face_detection, 'damo/cv_resnet101_face-detection_cvpr22papermogface')

    return model




def face_detection(model,model_name,img_batch):
    
    transform = transforms.ToTensor()
    mtcnn = MTCNN(
        image_size=128, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    loss = np.ones((len(img_batch),1))
    
    for i,img_batch_applied in enumerate(img_batch):
        img_batch_applied.save('image.jpg')
        #img_batch_applied = mtcnn(img_batch_applied,save_path='./scrfd:%i.jpg'%(i))
        
        if model_name == 'yolov5':  
            img_batch_applied = transform(img_batch_applied).unsqueeze(0)#.permute(0,3,1,2)
            img_batch_applied = F.interpolate(img_batch_applied,(128,128)).to(device)
            output = model.detector(img_batch_applied)[0] # (b,:,14)
            score = output[...,4]
            #score = torch.sigmoid(score)
            max_score = torch.max(score,dim=1)
            det_loss = torch.mean(max_score.values)   
        if model_name == 'yolov8':   
            score = model.detect(np.array(img_batch_applied)) 
            score = torch.from_numpy(np.array(score))
            #score = torch.sigmoid(score)
            max_score = torch.max(score)
            det_loss = torch.mean(max_score) 
            
        elif model_name == 'retinaface':
            img_batch_applied = transform(img_batch_applied).unsqueeze(0).to(device)
            det_loss = model.predict_jsons(img_batch_applied)
            
        elif model_name == 'dlib':
            sub_img_batch_applied = imutils.resize(np.array(img_batch_applied), width=300)
            # convert the image to grayscale
            sub_img_batch_applied = cv2.cvtColor(sub_img_batch_applied, cv2.COLOR_BGRA2GRAY)
            # detect faces in the image 
            rects = model(sub_img_batch_applied, upsample_num_times=1)    
            
            if len(rects) !=0:
                rect = rects[0]
                confidence = rect.confidence
                det_loss = torch.tensor(confidence)
                det_loss = det_loss.to(device)
            else:
                det_loss =  0
            
        elif model_name =='opencv_dnn':
            blob = cv2.dnn.blobFromImage(np.array(img_batch_applied),1.0,(112,112),[104,117,123],False,False)
            model.setInput(blob)
            detections = model.forward()
            bboxes = []
            ret = 0 
            max_score = torch.from_numpy(detections[:,:,:,2])
            #max_score = torch.sigmoid(max_score)
            det_loss = torch.max(max_score).to(device)
            
        elif model_name =='scrfd':
            result = model.get(np.array(img_batch_applied))
            score = [result[0]['det_score']]
            if len(score)!=0:
                det_loss = torch.max(torch.from_numpy(np.array(score))).to(device)
            else:
                det_loss = torch.tensor(0)
        elif model_name =='mtcnn':
            bboxes, landmarks = model.detect(img_batch_applied)
            
            if len(bboxes)!=0:
                score = bboxes[:,-1]
                if len(score)!=0:
                    det_loss = torch.max(torch.from_numpy(np.array(score))).to(device)
                else:
                    det_loss = torch.tensor(0)
            else:
                det_loss = torch.tensor(0)
            

        elif model_name =="dsfd":
            target_size = (128, 128)
            detections = model.detect_on_image(np.array(img_batch_applied), target_size, device, is_pad=False, keep_thresh=0.1)
            score = detections[:,-1]
            if len(score)!=0:
                det_loss = torch.max(torch.from_numpy(np.array(score))).to(device)
            else:
                det_loss = torch.tensor(0)
            
            

        elif model_name =="ulfd":
            result = model(img_batch_applied)
    
            scores = result['scores']
            
            if len(scores)!=0:
                det_loss = torch.max(torch.from_numpy(np.array(scores))).to(device)
            else:
                det_loss = torch.tensor(0)

        elif model_name =="mogface":
            img_batch_applied = transform(img_batch_applied).unsqueeze(0)
            img_batch_applied = F.interpolate(img_batch_applied,(416,416)).to(device)
            
            result = model(img_batch_applied)
            scores = torch.max(result[0].squeeze(0))
            det_loss = scores
            
            '''scores = result['scores']
            if len(scores)!=0:
                det_loss = torch.max(torch.from_numpy(np.array(scores))).to(device)
            else:
                det_loss = torch.tensor(0)'''
            
        #loss.append(det_loss.item())
        loss[i]=det_loss.item()
        loss = loss.reshape(-1)
       
        if det_loss <0.8:
            return loss
    
    return loss


