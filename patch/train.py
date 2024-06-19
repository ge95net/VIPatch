import sys
import os
    
import random
from pathlib import Path
import pickle
sys.path.append("/home/qiuchengyu/AdversarialMask") 
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import function
#import AdversarialMask.patch.function as function
import losses
from config import patch_config_types
from nn_modules import LandmarkExtractor, FaceXZooProjector, TotalVariation
from function import load_embedder, EarlyStopping, get_patch
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

import warnings
warnings.simplefilter('ignore', UserWarning)

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device is {}'.format(device), flush=True)


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

        self.train_loader = function.get_train_loaders(self.config)

        self.embedders = load_embedder(self.config.train_embedder_names, device)

        face_landmark_detector = function.get_landmark_detector(self.config, device)
        self.location_extractor = LandmarkExtractor(device, face_landmark_detector, self.config.img_size).to(device)
        self.fxz_projector = FaceXZooProjector(device, self.config.img_size, self.config.patch_size,config).to(device)
        self.total_variation = TotalVariation(device).to(device)
        self.dist_loss = losses.get_loss(self.config)

        self.train_losses_epoch = []
        self.train_losses_iter = []
        self.dist_losses = []
        self.tv_losses = []
        self.val_losses = []
        if self.config.detector =='yolov5':
            self.model = YoloDetector(target_size=720, device=device, min_face=90)
        elif self.config.detector =='yolov8':
            self.model =  YOLOv8_face('weights/yolov8n-face.onnx', conf_thres=self.config.threshold, iou_thres=0.4)
        elif  self.config.detector =='retinaface':
            self.model = get_model("resnet50_2020-07-20", max_size=2048)
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
            self.model = pipeline(Tasks.face_detection, 'damo/cv_resnet101_face-detection_cvpr22papermogface')
        self.create_folders()
        function.save_class_to_file(self.config, self.config.current_dir)
       
        self.best_patch = None

    def create_folders(self):
        Path('/'.join(self.config.current_dir.split('/')[:2])).mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/final_results/sim-boxes').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/final_results/pr-curves').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/final_results/stats/similarity').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/final_results/stats/average_precision').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/saved_preds').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/saved_patches').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/saved_similarities').mkdir(parents=True, exist_ok=True)
        Path(self.config.current_dir + '/losses').mkdir(parents=True, exist_ok=True)

    def train(self):
        adv_patch_cpu = function.get_patch(self.config)
        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)
        early_stop = EarlyStopping(current_dir=self.config.current_dir, patience=self.config.es_patience, init_patch=adv_patch_cpu)
        epoch_length = len(self.train_loader)
        for epoch in range(self.config.epochs):
            train_loss = 0.0
            dist_loss = 0.0
            tv_loss = 0.0
           
            for i_batch, img_batch in enumerate(self.train_loader):
                b_loss,dist_loss,tv_loss  = self.forward_step(img_batch, adv_patch_cpu)

                train_loss += b_loss.item()
                dist_loss += dist_loss.item()
                tv_loss += tv_loss.item()

                optimizer.zero_grad()
                b_loss.backward()
                optimizer.step()
                adv_patch_cpu.data.clamp_(0, 1)
            if early_stop(self.train_losses_epoch[-1], adv_patch_cpu, epoch):
                self.best_patch = adv_patch_cpu
                break

            scheduler.step(self.train_losses_epoch[-1])
        self.best_patch = early_stop.best_patch
        self.save_final_objects()
        function.plot_train_val_loss(self.config, self.train_losses_epoch, 'Epoch')
        function.plot_train_val_loss(self.config, self.train_losses_iter, 'Iterations')
        function.plot_separate_loss(self.config, self.train_losses_epoch, self.dist_losses, self.tv_losses)


    def forward_step(self, img_batch, adv_patch_cpu):
        img_batch = img_batch.to(device)
        adv_patch = adv_patch_cpu.to(device)
        
        preds = self.location_extractor(img_batch)
        img_batch_applied = self.fxz_projector(img_batch, preds, adv_patch, do_aug=self.config.mask_aug)
        save_image = img_batch_applied[0].permute(1,2,0).cpu().numpy()
        cv2.imwrite('best_image.jpg',save_image[:,:,::-1])
        output = self.model.detector(img_batch_applied)[0] # (b,:,14)
        score = output[...,4]
        #score = torch.sigmoid(score)
        max_score = torch.max(score,dim=1)
        det_loss = torch.mean(max_score.values)   
        print('det_loss=',det_loss)
        '''patch_embs = {}
        for embedder_name, emb_model in self.embedders.items():
            patch_embs[embedder_name] = emb_model(img_batch_applied)
        '''
        tv_loss = self.total_variation(adv_patch)
        #loss = self.loss_fn(patch_embs, tv_loss)
        loss = det_loss + tv_loss

        return loss,det_loss,tv_loss#, [img_batch, adv_patch, img_batch_applied, patch_embs, tv_loss]

    def save_losses(self, epoch_length, train_loss, dist_loss, tv_loss):
        train_loss /= epoch_length
        dist_loss /= epoch_length
        tv_loss /= epoch_length
        self.train_losses_epoch.append(train_loss)
        self.dist_losses.append(dist_loss)
        self.tv_losses.append(tv_loss)

    def save_final_objects(self):
        alpha = transforms.ToTensor()(Image.open('../prnet/new_uv.png').convert('L'))
        final_patch = torch.cat([self.best_patch.squeeze(0), alpha])
        final_patch_img = transforms.ToPILImage()(final_patch.squeeze(0))
        final_patch_img.save(self.config.current_dir + '/final_results/final_patch.png', 'PNG')
        new_size = tuple(self.config.magnification_ratio * s for s in self.config.img_size)
        transforms.Resize(new_size)(final_patch_img).save(self.config.current_dir + '/final_results/final_patch_magnified.png', 'PNG')
        torch.save(self.best_patch, self.config.current_dir + '/final_results/final_patch_raw.pt')

        with open(self.config.current_dir + '/losses/train_losses', 'wb') as fp:
            pickle.dump(self.train_losses_epoch, fp)
        with open(self.config.current_dir + '/losses/val_losses', 'wb') as fp:
            pickle.dump(self.val_losses, fp)
        with open(self.config.current_dir + '/losses/dist_losses', 'wb') as fp:
            pickle.dump(self.dist_losses, fp)
        with open(self.config.current_dir + '/losses/tv_losses', 'wb') as fp:
            pickle.dump(self.tv_losses, fp)


def main():
    mode = 'universal'
    config = patch_config_types[mode]()
    print('Starting train...', flush=True)
    adv_mask = AdversarialMask(config)
    adv_mask.train()
    print('Finished train...', flush=True)


if __name__ == '__main__':
    main()
