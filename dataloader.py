import torch
import numpy as np
from torch.utils.data import Dataset
import os 
import cv2
import copy
import torchvision.transforms as transforms
from PIL import Image
import math
import torch.nn.functional as F
from PIL import ImageFile



class image_Testdata(Dataset):
    def __init__(self,dir_path,config):
        super(image_Testdata).__init__()
        
        self.img = None
        
        self.all_path = []
        self.img_path = []
        self.dir_path = dir_path
        
        #self.lab_paths = os.listdir(self.dir_path)
        self.files = os.walk(self.dir_path)  

        for path,dir_list,file_list in self.files:  
            for file_name in file_list:  
                
                if file_name.endswith('jpg') or file_name.endswith('png') or file_name.endswith('JPG'):
                #if file_name.endswith('HEIC') :
                    file_path = os.path.join(path,file_name)
                
                    self.img_path.append(file_path)
                    if len(self.img_path) == 500:
                        break
            if len(self.img_path) == 500:
                break
        self.transform = transforms.Compose([transforms.Resize((128,128)),
                                               transforms.ToTensor()])
        
    def __getitem__(self, index):
        img_path = self.img_path[index]
        #img_path = 'datasets/thermal/gray/train/images/17_2_2_2_1107_45_1.png'
        #img_path = 'physical_images/examole_infrared.jpg'
        ###rgb
        img_path= '13.png'
        ###thermal
       
        image = Image.open(img_path)#.convert('RGB')
        image = self.transform(image).unsqueeze(0)
        
        
        return image
    
    def __len__(self):
        return len(self.img_path)
    


class Physical_image_Testdata(Dataset):
    def __init__(self,dir_path,config,model):
        super(Physical_image_Testdata).__init__()
        
        self.img = None
        self.model=model
        self.all_path = []
        self.img_path = []
        self.dir_path = dir_path
        
        #self.lab_paths = os.listdir(self.dir_path)
        self.files = os.walk(self.dir_path)  

        for path,dir_list,file_list in self.files:  
            for file_name in file_list:  
                
                if file_name.endswith('jpg') or file_name.endswith('png') or file_name.endswith('JPG') or file_name.endswith('bmp') or file_name.endswith('PNG'):
                #if file_name.endswith('HEIC') :
                    file_path = os.path.join(path,file_name)
                
                    self.img_path.append(file_path)
                    '''if len(self.img_path) == 5000:
                        break
            if len(self.img_path) == 5000:
                break'''
        self.target_size=832
        self.transform = transforms.Compose([
            #transforms.Resize((1120,1120)),
            transforms.Resize((416,416)),
            transforms.ToTensor()
        ])
        
    def __getitem__(self, index):
        
        image_path = self.img_path[index]
        print('image_path=',image_path)
        #image_path = 'datasets/physical_images/best_image3.bmp'
        print('img_path=',image_path)
        img = np.array(Image.open(image_path))
        #img = cv2.imread(image_path)
        if image_path == 'datasets/physical_images/IMG_1536.JPG' or image_path == 'datasets/physical_images/IMG_1551.JPG' :
            img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
        if self.model == 'yolo':
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            img = np.array(img)

            if img.ndim==2:
                img = np.expand_dims(img,2)
            
            if img.shape[2] == 1:
                img = np.repeat(img,repeats=3,axis=2)
                
            height = img.shape[0]
            width = img.shape[1]
            
            
            h0, w0 = img.shape[:2]  # orig hw
            if self.target_size:
                r = self.target_size / min(h0, w0)  # resize image to img_size
                if r < 1:  
                    img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
            
            imgsz = check_img_size(max(img.shape[:2]), s=self.model.detector.module.stride.max())  # check img_size
            img = letterbox(img, new_shape=imgsz)[0]
            
            h1 , w1 = img.shape[:2]
            original_size = [h0,w0]
            new_size = [h1,w1]
        img = Image.fromarray(img).convert('RGB')
        img = self.transform(img)
        
        img = img.float()
        
       

        
        return img,image_path
    
    def __len__(self):
        return len(self.img_path)
    
    
    

    
    
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



class image_Traindata(Dataset):
    def __init__(self,dir_path,model,max_n_labels):
        super(image_Traindata).__init__()
        
        self.img = None
        
        self.img_names = []
        self.img_path = []
        self.dir_path = dir_path
        
        
        self.files = os.walk(self.dir_path)  
        for path,dir_list,file_list in self.files:  
            for file_name in file_list:  
                if file_name.endswith('jpg') or file_name.endswith('png'):
                    file_path = os.path.join(path,file_name)
                    self.img_names.append(file_path)
                if len(self.img_names) == 500:
                    break
            if len(self.img_names) == 500:
                break
        
        self.target_size=416
        self.transform = transforms.Compose([
            transforms.Resize((self.target_size,self.target_size)),
            transforms.ToTensor()
        ])
        self.model = model
        self.max_n_labels = max_n_labels
        
    def __getitem__(self, index):
        
        image_path = self.img_names[index]
        #image_path = 'glass_image/example2.jpg'
        
        img = Image.open(image_path)
        
        
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = np.array(img)

        if img.ndim==2:
            img = np.expand_dims(img,2)
        
        if img.shape[2] == 1:
            img = np.repeat(img,repeats=3,axis=2)
            
        height = img.shape[0]
        width = img.shape[1]
        
        
        h0, w0 = img.shape[:2]  # orig hw
        if self.target_size:
            r = self.target_size / min(h0, w0)  # resize image to img_size
            if r < 1:  
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        
        imgsz = check_img_size(max(img.shape[:2]), s=self.model.detector.module.stride.max())  # check img_size
        img = letterbox(img, new_shape=imgsz)[0]
        
        h1 , w1 = img.shape[:2]
        original_size = [h0,w0]
        new_size = [h1,w1]
        img = Image.fromarray(img)
        img = self.transform(img)
        img = img.float()
        
        
        
        
       

        
        return img
    
      
    def __len__(self):
        return len(self.img_names)
    
    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if(pad_size>0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
        else:
            padded_lab = lab[:self.max_n_labels,:]
        
        return padded_lab
    
    def trans2yolo(self,label,original_size,new_size):
        label = label.numpy()
        new_label = []
        all_points = []
        for i in range(len(label)):
            new_sub_label = []
            sub_label = label[i]
            major_radius = sub_label[0] 
            minor_radius = sub_label[1]
            angle = sub_label[2]
            center_x = sub_label[3] * new_size[1]/original_size[1]
            center_y = sub_label[4] * new_size[0]/original_size[0]
            A, B, C, F = get_ellipse_param(major_radius, minor_radius, angle)
            p1, p2 = calculate_rectangle(A, B, C, F)
            rectangle_height = abs(p2[1]) + abs(p2[0])
            rectangle_width = abs(p1[1]) + abs(p1[0])
           
            new_sub_label.append(int(0))
            new_sub_label.append((center_x/new_size[1]).astype(np.float32))
            new_sub_label.append((center_y/new_size[0]).astype(np.float32))
            new_sub_label.append((rectangle_width/new_size[1]).astype(np.float32))
            new_sub_label.append((rectangle_height/new_size[0]).astype(np.float32))
            new_label.append(new_sub_label)
            
            points = [center_x+p1[0], center_y+p1[1], center_x+p2[0], center_y+p2[1],center_x,center_y]
            all_points.append(points)
        return torch.from_numpy(np.array(new_label)), all_points