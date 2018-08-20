# in this file I will encapsualte the necessary functions for doing the Rahan Image jobs (and related)
# 1. train classifier from file lists
# 2. train detector

#### Classifier #####

# multie use generator, tested
import json
import os
import sys
import glob
from sklearn.model_selection import train_test_split
# imports
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import os
import sys
#from utils import * # phenmoics utils
from imgaug import augmenters as iaa

ROOT_DIR = os.path.abspath("/home/gidish/cloned_libs/Mask_RCNN/") # add here mask RCNN path
sys.path.append(ROOT_DIR)  # To find local version of the library
model_path='/mnt/gluster/home/gidish/models'

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

from datetime import datetime

import glob
from sklearn.model_selection import train_test_split
import skimage


import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import imgaug
import PIL
# Root directory of the project

# Import Mask RCNN

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from samples.coco import coco
from samples.balloon import balloon


# this MRCNN requires dataset class
# most of this class is similar to coco. can be inherited
class FlowerCocoDataset(coco.CocoDataset):
    
    def load_flowers(self, dataset_dirs, subset, subset_files='', class_ids='', return_coco=False, correction=False):
        self.correction =correction 
        # gidi: correction is our addition if segmentation format is 2 
        # lists instead of 1

        #coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        for dataset_dir in dataset_dirs:

            coco = COCO(dataset_dir+'segmentation_results.json')

            if not class_ids:
                # All classes
                class_ids = sorted(coco.getCatIds())

            # Add classes
            for i in class_ids:
                self.add_class("coco", i, coco.loadCats(i)[0]["name"])

            # Add images

            for i in coco.imgs:
                if coco.imgs[i]['path'] in subset_files:
                    self.add_image(
                        "coco", image_id=i,
                        path=os.path.join(dataset_dir, coco.imgs[i]['file_name']),
                        width=coco.imgs[i]["width"],
                        height=coco.imgs[i]["height"],
                        annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None)))

        if return_coco:
            return coco

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        
        zipped=[]
        #print(ann)
        
        if len(ann) > 1 and self.correction:
            ann = np.array(ann['segmentation'])
            n = len(ann[0])
            ann_reshaped = np.reshape(ann, 2*n ,order='F')
            
            new_ann = [ann_reshaped.tolist()]
            segm = new_ann
        else:
            segm = ann['segmentation']

        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif 'counts' in segm:
            if isinstance(segm['counts'], list):
                # uncompressed RLE
                rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann
        return rle
    
# see that files are jpg and not JPG
class ViaDataset(utils.Dataset):

    def load_flowers(self, dataset_dirs,subset_files, subset='train'):
        # Gidi: instead of dir, I lod train test sets by myself 
        # subset files is the thing
        """Load a subset of the >< dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("flower", 1, "flower")
        for dataset_dir in dataset_dirs:
            annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
            annotations = list(annotations.values())  # don't need the dict keys

            # The VIA tool saves images in the JSON even if they don't have any
            # annotations. Skip unannotated images.
            annotations = [a for a in annotations if a['regions']]

            # Add images
            for a in annotations:
                # Get the x, y coordinaets of points of the polygons that make up
                # the outline of each object instance. There are stores in the
                # shape_attributes (see json format above)
                polygons = [r['shape_attributes'] for r in a['regions'].values()]

                # load_mask() needs the image size to convert polygons to masks.
                # Unfortunately, VIA doesn't include it in JSON, so we must read
                # the image. This is only managable since the dataset is tiny.
                if a['filename'] in subset_files:
                    image_path = os.path.join(dataset_dir, a['filename'])
                    image = PIL.Image.open(image_path)
                    height, width = image.size[::-1]
                    self.add_image("flower",image_id=a['filename'], path=image_path,width=width, height=height,polygons=polygons)# use file name as a unique image id

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            if p['name']=='polygon':
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                mask[rr, cc, i] = 1
            if p['name'] == 'circle':
                rr, cc = skimage.draw.circle(p['cy'], p['cx'], p['r'])
                # the folloing row is for out of bounds circles
                rr, cc = np.array([[row, col] for row, col in zip(rr, cc) if row < info["height"] and col < info["width"] and row>0 and col>0]).T
                mask[rr, cc, i] = 1
            if p['name'] == 'rect':
        
                rr, cc = skimage.draw.rectangle([p['y'], p['x']], [p['y']+p['height'], p['x']+p['width']])
                #rr, cc = np.array([[row,col] for row,col in zip(rr,cc) if row<info["height"] and col<info["width"]]).T
                mask[rr, cc, i] = 1
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

from mrcnn.config import Config

class FlowersConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
        
    # Give the configuration a recognizable name
    NAME = ''
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 512 # larger images for small flowers like avocado

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

class InferenceConfig(FlowersConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
config = FlowersConfig()
inference_config = InferenceConfig()

def generator_from_list(full_list, bs=16):
    while True:
        x=[]
        y=[]
        classes= ['banana','avocado','cucumber','tomato']
        l=np.random.choice(len(full_list),bs)
        for file in np.array(full_list)[l]:
            try:
                x.append(resize(np.array(Image.open(file)),[224,224])[:,:,:3])
                y.append(np.argmax([clss in file for clss in classes])) # return number
            except:
                pass
                #print('error in loading image',file)
        x=np.array(x)
        y=np.array(y)
        yield (x, keras.utils.to_categorical(y,len(classes)))

# val generator (not tested)
def generator_val_list(val_list):
    while True:
        

        x=[]
        y=[]
        classes= ['banana','avocado','cucumber','tomato']
        for file in val_list:
            try:
                x.append(resize(np.array(Image.open(file)),[224,224])[:,:,:3])
                y.append(np.argmax([clss in file for clss in classes])) # return number
            except:
                pass
                #print('error in loading image',file)
        x=np.array(x)
        y=np.array(y)
        yield (x, keras.utils.to_categorical(y,len(classes)))
        

data_types ={'banana_flowers':'via','avocado_flowers':'via','cucumber_flowers_3':'coco'}


def banana_compute_ap(gt_boxes, gt_class_ids, gt_masks,pred_boxes, pred_class_ids, pred_scores, pred_masks):
    gt_match, pred_match, overlaps = utils.compute_matches(gt_boxes, gt_class_ids, gt_masks,pred_boxes, pred_class_ids, pred_scores, pred_masks)
    # compute AP for special banana case
    # clean redundant detectio
    
    detect=0
    for i,o in enumerate(overlaps):

        if detect==1 and o>0.5:
            overlaps=np.delete(overlaps,i)
            pred_match=np.delete(pred_match,i)
            gt_match--1
        if o>0.5:
            detect=1
    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *precisions[indices])
    return mAP

class Flower_detection:
    def __init__(self, task):
        self.task = task
    
    def load_data(self, data_type):
        #data_type: coco/via
        if 'cucumber' in self.task:
            dataset_dirs = [ f'/home/gidish/data/Rahan/cucumber_flowers/', f'/home/gidish/data/Rahan/cucumber_flowers_3/']
        else:
            dataset_dirs = [ f'/home/gidish/data/Rahan/{self.task}//']
        
        files=[]  
        for dir in dataset_dirs:
            files+=glob.glob(dir+'/*.jpg')+glob.glob(dir+'/*.JPG')+glob.glob(dir+'/*.png')
        #full_train_files, val_files = train_test_split(files,random_state=1)
        files = [file.split("/")[-1] for file in files]
        train_files,val_files = train_test_split(files,random_state=1)
        print(f'loading {self.task} data')
        if data_type=='coco':

            if  len(train_files)>30:
                self.train_dataset = FlowerCocoDataset()
                self.train_dataset.load_flowers(dataset_dirs,'',train_files,correction=False) # pay attention to correction...
                self.train_dataset.prepare()

                self.val_dataset = FlowerCocoDataset()
                self.val_dataset.load_flowers(dataset_dirs,'',val_files,correction=False) # pay attention to correction...
                self.val_dataset.prepare()
            else:
                dataset = FlowerCocoDataset()
                dataset.load_flowers(dataset_dirs,'',train_files,correction=True) # pay attention to correction...
                dataset.prepare()
                self.train_dataset=self.val_dataset=dataset


        # perhaps image size should be/can be saves somehow to be loaded later
        elif data_type=='via':
            self.val_dataset = ViaDataset()
            self.val_dataset.load_flowers(dataset_dirs, val_files,'val')
            self.val_dataset.prepare()

            self.train_dataset = ViaDataset()
            self.train_dataset.load_flowers(dataset_dirs, train_files)
            self.train_dataset.prepare()
            # Must call before using the dataset

        self.test_dataset = self.val_dataset
        print("Images: {}\nClasses: {}".format(len(self.train_dataset.image_ids), self.train_dataset.class_names))
        print("Images: {}\nClasses: {}".format(len(self.val_dataset.image_ids), self.val_dataset.class_names))

    def load_model(self, init_with, mode = 'training'):
        #init with: last/coco

        if mode=='inference':
            print('inference mode')
            inference_config.NAME = self.task
            self.model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

        elif mode == 'training':
            print('training mode')
            config.NAME = self.task
            self.model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

        if init_with == "imagenet":
            self.model.load_weights(model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            COCO_MODEL_PATH = os.path.join(model_path, "mask_rcnn_coco.h5")
            self.model.load_weights(COCO_MODEL_PATH, by_name=True,exclude=["mrcnn_class_logits","mrcnn_bbox_fc","mrcnn_bbox","mrcnn_mask"])
        elif init_with == "last":
            # Load the last model you trained and continue training
            print(f'load model from {self.model.find_last()}')
            self.model.load_weights(self.model.find_last(), by_name=True) #[1]
        return self.model

    # main eval function
    def evaluate(self, n=10):
        # evaluate mAP for crop.
        # version can be '' or banana
        image_ids = self.test_dataset.image_ids# np.random.choice(dataset.image_ids, 10)

        APs = []
        for image_id in image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(self.test_dataset, inference_config, image_id, use_mini_mask=False)
            # molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
            # Run object detection
            results = self.model.detect([image], verbose=0)
            r = results[0]
            #visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],  class_names, r['scores'], ax=get_ax())
            # Compute AP
            try:
                if 'banana' in self.task:
                    AP = banana_compute_ap(gt_bbox, gt_class_id, gt_mask,r["rois"], r["class_ids"], r["scores"], r['masks'])
                else:
                    AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,r["rois"], r["class_ids"], r["scores"], r['masks'])
            except:
                AP=0
                print('error in compute')
            APs.append(AP)

        print("mAP: ", np.mean(APs))
        return AP, APs

    def load_data_model_results(self, mode = 'inference', metric ='map'):
        # load data
        self.load_data(data_types[self.task])
        print(mode)
        if mode == 'inference':
            self.model = self.load_model('last', mode)
        if mode == 'training':
            self.model = self.load_model('coco', mode)
            self.model.train(self.train_dataset, self.val_dataset, learning_rate=config.LEARNING_RATE, epochs=30,  layers='heads',
                         augmentation = seq)
        # calc metric
        if metric=='map':
            self.evaluate()





        