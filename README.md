# VIPatch: Physical Adversarial Patch Attacks against Visual-Infrared Fused Face Detection Systems



## setup
```
1.git clone git@github.com:ge95net/VIPatch.git

2.pip install -r requirements.txt
```


## Face Detection Model
1, Yolov5-Face

Download the checkpoints:  [Yolov5-Face](https://github.com/deepcam-cn/yolov5-face#pretrained-models).

put it in the 'weights/'

2, Yolov8-Face

Download the checkpoints: [Yolov8-Face](https://github.com/derronqi/yolov8-face).

put it in the 'weights/'

3,OpenCV

Download the checkpoints: opencv_face_detector.pbtxt and opencv_face_detector_uint8.pb from [OpenCV](https://github.com/spmallick/learnopencv/tree/master/AgeGender).

Put it in the 'models/'

4, Retinaface

Following the instruction of [Retinaface](https://github.com/serengil/retinaface)

5, Mogface
Download the checkpoints: [Mogface](https://github.com/damo-cv/MogFace?tab=readme-ov-file)

put it in the ''models/MogFace/configs/mogface/'

## Image Harmonization

Follow the instruction of [libcom](https://github.com/bcmi/libcom?tab=readme-ov-file) to install the toolbox.


## Sticker
Require:
(1) BFM
Following the instruction of [Face3D](https://github.com/yfeng95/face3d) to Download the BFM.dat

put it in the 'models/BFM/'

(2)Shape predictor for face landmarks ([68](https://github.com/r4onlyrishabh/facial-detection/tree/master/dataset), [81](https://github.com/codeniko/shape_predictor_81_face_landmarks))

## Data Preparation

Download the dataset from [Speaking Faces](https://issai.nu.edu.kz/speaking-faces/)

The directory structure example is:
'''
datasets

  -thermal
  
    --gray
    
      ---train
      
        ----pic001
        
        ----pic002
        
      ---test
      
        ----pic001
        
        ----pic002
        
    --rgb
    
      ---train
      
        ----pic001
        
        ----pic002
        
      ---test
      
        ----pic001
        
        ----pic002
        
'''



