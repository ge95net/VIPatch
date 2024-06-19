import joblib
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import copy
import scipy
import pathlib
import warnings
import math
from math import sqrt
sys.path.append('/home/qiuchengyu/AdversarialMask')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
from models.common import Conv
from models.yolo import Model
from models.utils.datasets import letterbox
from models.utils.preprocess_utils import align_faces
from models.utils.general import check_img_size, non_max_suppression_face, \
    scale_coords,scale_coords_landmarks,filter_boxes

class YoloDetector:
    def __init__(self, weights_name='yolov5n_state_dict.pt', config_name='yolov5n.yaml', device='cuda:0', min_face=100,cuda_num='0', target_size=None, frontal=False):
            """
            weights_name: name of file with network weights in weights/ folder.
            config_name: name of .yaml config with network configuration from models/ folder.
            device : pytorch device. Use 'cuda:0', 'cuda:1', e.t.c to use gpu or 'cpu' to use cpu.
            min_face : minimal face size in pixels.
            target_size : target size of smaller image axis (choose lower for faster work). e.g. 480, 720, 1080. Choose None for original resolution.
            frontal : if True tries to filter nonfrontal faces by keypoints location. CURRENTRLY UNSUPPORTED.
            """
            self._class_path = pathlib.Path(__file__).parent.absolute()#os.path.dirname(inspect.getfile(self.__class__))
            self.device = device
            self.target_size = target_size
            self.min_face = min_face
            self.frontal = frontal
            if self.frontal:
                print('Currently unavailable')
                # self.anti_profile = joblib.load(os.path.join(self._class_path, 'models/anti_profile/anti_profile_xgb_new.pkl'))
            self.detector = self.init_detector(weights_name,config_name,cuda_num)

    def init_detector(self,weights_name,config_name,cuda_num):
        
        model_path = os.path.join(self._class_path,'weights/',weights_name)
        
        config_path = os.path.join(self._class_path,'models/',config_name)
        state_dict = torch.load(model_path)
        detector = Model(cfg=config_path)
        detector.load_state_dict(state_dict)
        
        detector = nn.DataParallel(detector,device_ids=[4])
        detector = detector.to(self.device).float().eval()
        
        for m in detector.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        return detector
    
    def _preprocess(self,imgs):
        """
            Preprocessing image before passing through the network. Resize and conversion to torch tensor.
        """
        pp_imgs = []
        for img in imgs:
            h0, w0 = img.shape[:2]  # orig hw
            if self.target_size:
                r = self.target_size / min(h0, w0)  # resize image to img_size
                if r < 1:  
                    img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)

            imgsz = check_img_size(max(img.shape[:2]), s=self.detector.stride.max())  # check img_size
            img = letterbox(img, new_shape=imgsz)[0]
            pp_imgs.append(img)
        pp_imgs = np.array(pp_imgs)
        pp_imgs = pp_imgs.transpose(0, 3, 1, 2)
        pp_imgs = torch.from_numpy(pp_imgs).to(self.device)
        pp_imgs = pp_imgs.float()  # uint8 to fp16/32
        pp_imgs /= 255.0  # 0 - 255 to 0.0 - 1.0
        return pp_imgs
        
    def _postprocess(self, imgs, origimgs, pred, conf_thres, iou_thres):
        """
            Postprocessing of raw pytorch model output.
            Returns:
                bboxes: list of arrays with 4 coordinates of bounding boxes with format x1,y1,x2,y2.
                points: list of arrays with coordinates of 5 facial keypoints (eyes, nose, lips corners).
        """
        bboxes = [[] for i in range(len(origimgs))]
        landmarks = [[] for i in range(len(origimgs))]
        
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)
        
        for i in range(len(origimgs)):
            img_shape = origimgs[i].shape
            h,w = img_shape[:2]
            gn = torch.tensor(img_shape)[[1, 0, 1, 0]]  # normalization gain whwh
            gn_lks = torch.tensor(img_shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]  # normalization gain landmarks
            det = pred[i].cpu()
            scaled_bboxes = scale_coords(imgs[i].shape[1:], det[:, :4], img_shape).round()
            scaled_cords = scale_coords_landmarks(imgs[i].shape[1:], det[:, 5:15], img_shape).round()

            for j in range(det.size()[0]):
                box = (det[j, :4].view(1, 4) / gn).view(-1).tolist()
                box = list(map(int,[box[0]*w,box[1]*h,box[2]*w,box[3]*h]))
                if box[3] - box[1] < self.min_face:
                    continue
                lm = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                lm = list(map(int,[i*w if j%2==0 else i*h for j,i in enumerate(lm)]))
                lm = [lm[i:i+2] for i in range(0,len(lm),2)]
                bboxes[i].append(box)
                landmarks[i].append(lm)
        return bboxes, landmarks

    def get_frontal_predict(self, box, points):
        '''
            Make a decision whether face is frontal by keypoints.
            Returns:
                True if face is frontal, False otherwise.
        '''
        cur_points = points.astype('int')
        x1, y1, x2, y2 = box[0:4]
        w = x2-x1
        h = y2-y1
        diag = sqrt(w**2+h**2)
        dist = scipy.spatial.distance.pdist(cur_points)/diag
        predict = self.anti_profile.predict(dist.reshape(1, -1))[0]
        if predict == 0:
            return True
        else:
            return False
    def align(self, img, points):
        '''
            Align faces, found on images.
            Params:
                img: Single image, used in predict method.
                points: list of keypoints, produced in predict method.
            Returns:
                crops: list of croped and aligned faces of shape (112,112,3).
        '''
        crops = [align_faces(img,landmark=np.array(i)) for i in points]
        return crops

    def predict(self, imgs, conf_thres = 0.3, iou_thres = 0.5):
        '''
            Get bbox coordinates and keypoints of faces on original image.
            Params:
                imgs: image or list of images to detect faces on
                conf_thres: confidence threshold for each prediction
                iou_thres: threshold for NMS (filtering of intersecting bboxes)
            Returns:
                bboxes: list of arrays with 4 coordinates of bounding boxes with format x1,y1,x2,y2.
                points: list of arrays with coordinates of 5 facial keypoints (eyes, nose, lips corners).
        '''
        one_by_one = False
        # Pass input images through face detector
        if type(imgs) != list:
            images = [imgs]
        else:
            images = imgs
            one_by_one = False
            shapes = {arr.shape for arr in images}
            if len(shapes) != 1:
                one_by_one = True
                warnings.warn(f"Can't use batch predict due to different shapes of input images. Using one by one strategy.")
        origimgs = copy.deepcopy(images)
        
        
        if one_by_one:
            images = [self._preprocess([img]) for img in images]
            bboxes = [[] for i in range(len(origimgs))]
            points = [[] for i in range(len(origimgs))]
            for num, img in enumerate(images):
                with torch.inference_mode(): # change this with torch.no_grad() for pytorch <1.8 compatibility
                    single_pred = self.detector(img)[0]
                    print(single_pred.shape)
                bb, pt = self._postprocess(img, [origimgs[num]], single_pred, conf_thres, iou_thres)
                #print(bb)
                bboxes[num] = bb[0]
                points[num] = pt[0]
        else:
            images = self._preprocess(images)
            with torch.inference_mode(): # change this with torch.no_grad() for pytorch <1.8 compatibility
                pred = self.detector(images)[0]
            bboxes, points = self._postprocess(images, origimgs, pred, conf_thres, iou_thres)

        return bboxes, points

    def __call__(self,*args):
        return self.predict(*args)
    



class YOLOv8_face:
    def __init__(self, path, conf_thres=0.2, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.class_names = ['face']
        self.num_classes = len(self.class_names)
        # Initialize model
        self.net = cv2.dnn.readNet(path)
        self.input_height = 128
        self.input_width = 128
        self.reg_max = 16

        self.project = np.arange(self.reg_max)
        self.strides = (8, 16, 32)
        self.feats_hw = [(math.ceil(self.input_height / self.strides[i]), math.ceil(self.input_width / self.strides[i])) for i in range(len(self.strides))]
        self.anchors = self.make_anchors(self.feats_hw)

    def make_anchors(self, feats_hw, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points = {}
        for i, stride in enumerate(self.strides):
            h,w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset  # shift x
            y = np.arange(0, h) + grid_cell_offset  # shift y
            sx, sy = np.meshgrid(x, y)
            # sy, sx = np.meshgrid(y, x)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points

    def softmax(self, x, axis=1):
        x_exp = np.exp(x)
        # 如果是列向量，则axis=0
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s
    
    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_width, self.input_height
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_height, int(self.input_width / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_width - neww - left, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))  # add border
            else:
                newh, neww = int(self.input_height * hw_scale), self.input_width
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_height - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_height - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))
        else:
            img = cv2.resize(srcimg, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def detect(self, srcimg):
        input_img, newh, neww, padh, padw = self.resize_image(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB))
        scale_h, scale_w = srcimg.shape[0]/newh, srcimg.shape[1]/neww
        input_img = input_img.astype(np.float32) / 255.0

        blob = cv2.dnn.blobFromImage(input_img)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        
        # if isinstance(outputs, tuple):
        #     outputs = list(outputs)
        # if float(cv2.__version__[:3])>=4.7:
        #     outputs = [outputs[2], outputs[0], outputs[1]] ###opencv4.7需要这一步，opencv4.5不需要
        # Perform inference on the image
        scores = self.post_process(outputs, scale_h, scale_w, padh, padw)
        return scores

    def post_process(self, preds, scale_h, scale_w, padh, padw):
        bboxes, scores, landmarks = [], [], []
        for i, pred in enumerate(preds):
            stride = int(self.input_height/pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))
            
            box = pred[..., :self.reg_max * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1,1))
            kpts = pred[..., -15:].reshape((-1,15)) ### x1,y1,score1, ..., x5,y5,score5

            # tmp = box.reshape(self.feats_hw[i][0], self.feats_hw[i][1], 4, self.reg_max)
            tmp = box.reshape(-1, 4, self.reg_max)
            bbox_pred = self.softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1,4))

            bbox = self.distance2bbox(self.anchors[stride], bbox_pred, max_shape=(self.input_height, self.input_width)) * stride
            kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (self.anchors[stride][:, 0].reshape((-1,1)) - 0.5)) * stride
            kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (self.anchors[stride][:, 1].reshape((-1,1)) - 0.5)) * stride
            kpts[:, 2::3] = 1 / (1+np.exp(-kpts[:, 2::3]))

            bbox -= np.array([[padw, padh, padw, padh]])  ###合理使用广播法则
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])
            kpts -= np.tile(np.array([padw, padh, 0]), 5).reshape((1,15))
            kpts *= np.tile(np.array([scale_w, scale_h, 1]), 5).reshape((1,15))

            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)
    
        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  ####xywh
        classIds = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)  ####max_class_confidence
        
        mask = confidences>self.conf_threshold
        bboxes_wh = bboxes_wh[mask]  ###合理使用广播法则
        confidences = confidences[mask]
        classIds = classIds[mask]
        landmarks = landmarks[mask]
        return scores
        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.conf_threshold,
                                   self.iou_threshold).flatten()
        if len(indices) > 0:
            mlvl_bboxes = bboxes_wh[indices]
            confidences = confidences[indices]
            classIds = classIds[indices]
            landmarks = landmarks[indices]
            return mlvl_bboxes, confidences, classIds, landmarks
        else:
            print('nothing detect')
            return np.array([]), np.array([]), np.array([]), np.array([])

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    def draw_detections(self, image, boxes, scores, kpts):
        for box, score, kp in zip(boxes, scores, kpts):
            x, y, w, h = box.astype(int)
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
            cv2.putText(image, "face:"+str(round(score,2)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
            for i in range(5):
                cv2.circle(image, (int(kp[i * 3]), int(kp[i * 3 + 1])), 4, (0, 255, 0), thickness=-1)
                # cv2.putText(image, str(i), (int(kp[i * 3]), int(kp[i * 3 + 1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=1)
        return image

if __name__=='__main__':
    a = YoloDetector()
