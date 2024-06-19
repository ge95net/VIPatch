# VIPatch: Physical Adversarial Patch Attacks against Visual-Infrared Fused Face Detection Systems

## Abstract
Deep learning-based visual-infrared fused face detection models are increasingly utilized across various applications but have been shown to be susceptible to adversarial patch attacks. Most previous attacks have primarily targeted either the visual or infrared image alone in the digital domain, proving ineffective against visual-infrared fused face detection models in the physical world, and most of these existing methods are suspicious as their patched pattern significantly deviates from real-world patterns. In this paper, we introduce a novel physical adversarial patch attack, named VIPatch (Visual-Infrared Patch), which produces inconspicuous, realistic, and natural-looking patches for facial images. Specifically, this is accomplished by creating a gradient color mask and applying a band-aid sticker across both the visual and infrared images, with joint optimization of these two elements, and the generated digital patches could also guide the creation of their physical patches. Experimental results show that our method achieves competitive overall attack success rates (over 90%) in both digital and physical domains, and the patches created by our method are designed to attract minimal attention.

<img src="https://github.com/ge95net/VIPatch/example/demo.png" width="500" />

## setup
```
1.git clone git@github.com:ge95net/VIPatch.git

2, cd VIPatch

3.pip install -r requirements.txt
```


## Face Detection Model
1, Yolov5-Face

Download the checkpoints:  [Yolov5-Face](https://github.com/deepcam-cn/yolov5-face#pretrained-models).

put it in the 'weights/'.

2, Yolov8-Face

Download the checkpoints: [Yolov8-Face](https://github.com/derronqi/yolov8-face).

put it in the 'weights/'.

3,OpenCV

Download the checkpoints: opencv_face_detector.pbtxt and opencv_face_detector_uint8.pb from [OpenCV](https://github.com/spmallick/learnopencv/tree/master/AgeGender).

Put it in the 'models/'.

4, Retinaface

Following the instruction of [Retinaface](https://github.com/serengil/retinaface).

5, Mogface
Download the checkpoints: [Mogface](https://github.com/damo-cv/MogFace?tab=readme-ov-file).

put it in the ''models/MogFace/configs/mogface/'.

## Image Harmonization

Follow the instruction of [libcom](https://github.com/bcmi/libcom?tab=readme-ov-file) to install the toolbox.


## Sticker
Require:
(1) BFM
Following the instruction of [Face3D](https://github.com/yfeng95/face3d) to Download the BFM.dat.

put it in the 'models/BFM/'.

(2)Shape predictor for face landmarks ([68](https://github.com/r4onlyrishabh/facial-detection/tree/master/dataset), [81](https://github.com/codeniko/shape_predictor_81_face_landmarks)).

## Data Preparation

Download the dataset from [Speaking Faces](https://issai.nu.edu.kz/speaking-faces/) and put it in the folder 'datasets/'.

The directory structure example is:

```
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
```

## Quick Start
```
python patch/black_box_attack.py
```


## Change Parameters

You can change every parameter in the 'patch/config.py'
