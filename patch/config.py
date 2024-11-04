from torch import optim
import os
import datetime
import time




class BaseConfiguration:
    def __init__(self):
        self.seed = 42
        self.n_queries = 10000
        self.p_init = 0.4
        self.threshold = 0
        self.visual_threshold = 0.83 # yolov5=0.83, yolov8,opencv_dnn_scrfd,scrfd = 0.8, mtcnn,retinaface,ulfd = 0.97 #  These thresholds are kept consistent across all comparison methods.
        self.thermal_threshold = 0.6
        self.num_aug = 1
        self.patch_name = 'base'
        self.detector = 'yolov5' # yolov5,yolov8,mtcnn,retinaface,opencv_dnn,scrfd,,ulfd
        self.visual_detector = 'yolov5' # yolov5,yolov8,retinaface,dlib,opencv_dnn,scrfd,mtcnn,ulfd,mogface
        self.thermal_detector = 'yolov5' # yolov5,yolov8,mtcnn,retinaface,opencv_dnn,scrfd,,ulfd
        self.infrared_method = 'evo_diff'  # random,mask_block, evo_diff
        self.num_block = 20
        self.best_num_block = 4
        self.length = 0.4
        self.mode = 'train'   # train , test_thermal, test_visual, test_defense
        self.defense_mode = 'median_filter' #lgs, JPEG, median_filter,gaussian_filter, bit_depth_reduction
        self.rectangle_block = True
        self.do_aug = True
        self.harmonic_color ='direct' #analogous ,direct
        # Train dataset options
        self.is_real_person = False
        #self.train_dataset_name = 'right_face_70_yol
        # o'#'adversarialMask2'#'adversarialMask'  #noMask  normalMask
        self.visual_img_dir ='thermal/rgb/train'#physical2/distance/3/'#'#thermal/rgb/train'#'rs/angle/90/'#'physical2/light/12000/'#'thermal/rgb/train'#
        self.thermal_img_dir = 'thermal/gray/train/images'#'hotcold','cold_block_physical'#'thermal_noMask'#'thermal/gray/train/images'
        
        self.visual_img_dir = os.path.join('datasets', self.visual_img_dir)
        self.thermal_img_dir = os.path.join('datasets', self.thermal_img_dir)
  
        self.train_number_of_people = 100
        #self.celeb_lab = os.listdir(self.train_img_dir)[:self.train_number_of_people]
        
        #self.celeb_lab_mapper = {i: lab for i, lab in enumerate(self.celeb_lab)}
        self.num_of_train_images = 5
        self.patch_num=30
        self.shuffle = True
        self.img_size = (128, 128)
        self.train_batch_size = 4
        self.test_batch_size = 32
        self.magnification_ratio = 35

        # Attack options
        self.mask_aug = False
        self.patch_size = (128, 128)  # height, width
        self.initial_patch = 'random'  # body, white, random, stripes, l_stripes
        self.epochs = 10000
        self.start_learning_rate = 1e-2
        self.es_patience = 7
        self.sc_patience = 100
        self.sc_min_lr = 1e-6
        self.scheduler_factory = lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                                        patience=self.sc_patience,
                                                                                        min_lr=self.sc_min_lr,
                                                                                        mode='min')

        # Landmark detection options
        self.landmark_detector_type ='mobilefacenet'  # face_alignment, mobilefacenet

        # Embedder options
        self.train_embedder_names = []#['resnet100_arcface', 'resnet100_cosface', 'resnet100_magface']
        self.test_embedder_names = ['resnet100_arcface', 'resnet100_cosface', 'resnet100_magface']
        '''['resnet100_arcface', 'resnet50_arcface', 'resnet34_arcface', 'resnet18_arcface',
                                    'resnet100_cosface', 'resnet50_cosface', 'resnet34_cosface', 'resnet18_cosface',
                                    'resnet100_magface']'''

        # Loss options
        self.dist_loss_type = 'cossim'
        self.dist_weight = 1
        self.tv_weight = 0.1

        # Test options
        self.masks_path = os.path.join( 'data', 'masks')
        self.random_mask_path = os.path.join(self.masks_path, 'mask_random.png')
        self.blue_mask_path = os.path.join(self.masks_path, 'blue.png')
        self.black_mask_path = os.path.join(self.masks_path, 'black.png')
        self.white_mask_path = os.path.join(self.masks_path, 'white.png')
        self.face1_mask_path = os.path.join(self.masks_path, 'face1.png')
        self.face3_mask_path = os.path.join(self.masks_path, 'face3.png')

        self.update_current_dir()

    def set_attribute(self, name, value):
        setattr(self, name, value)

    def update_current_dir(self):
        my_date = datetime.datetime.now()
        month_name = my_date.strftime("%B")
        self.current_dir = os.path.join("experiments", month_name, time.strftime("%d-%m-%Y") + '_' + time.strftime("%H-%M-%S"))
        if 'SLURM_JOBID' in os.environ.keys():
            self.current_dir += '_' + os.environ['SLURM_JOBID']


class UniversalAttack(BaseConfiguration):
    def __init__(self):
        super(UniversalAttack, self).__init__()
        self.patch_name = 'universal'
        self.num_of_train_images = 5
        self.train_batch_size = 1
        self.test_batch_size = 32

        # Test dataset options
        self.test_num_of_images_for_emb = 5
        self.test_dataset_names = ['CASIA']
        self.test_img_dir = {name: os.path.join('datasets', name) for name in self.test_dataset_names}
        self.test_number_of_people = 200
        self.test_celeb_lab = {}
        


class TargetedAttack(BaseConfiguration):
    def __init__(self):
        super(TargetedAttack, self).__init__()
        self.patch_name = 'targeted'
        self.num_of_train_images = 10
        self.train_batch_size = 1
        self.test_batch_size = 4
        self.test_img_dir = {self.train_dataset_name: self.train_img_dir}




patch_config_types = {
    "base": BaseConfiguration,
    "universal": UniversalAttack,
    "targeted": TargetedAttack,
}
