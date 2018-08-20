# Written by ShiGidi

#imports
from utils_imports import *

from mrcnn.config import Config

def rleToMask(rleString,height,width):
    rows,cols = height,width
    rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
    rlePairs = np.array(rleNumbers).reshape(-1,2)
    img = np.zeros(rows*cols,dtype=np.uint8)
    for index,length in rlePairs:
        index -= 1
        img[index:index+length] = 255
    img = img.reshape(cols,rows)
    img = img.T
    return img

def get_aug():
    aug = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,iaa.GaussianBlur(sigma=(0, 0.5))),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-15, 15),
        shear=(-8, 8)
    )
], random_order=True)
    return aug

class RLEDataset(utils.Dataset):

    # actually this is init images
    def load_images(self, dataset_dirs, subset_files, subset='train',image_size=[768,768]):
        # Gidi: instead of dir, I lod train test sets by myself 
        # image_size='load', or tuple

        self.add_class("ship", 1, "ship")
        for dataset_dir in dataset_dirs:
            annotations = pd.read_csv(os.path.join(dataset_dir, "train_ship_segmentations.csv"))
            files = list(annotations.ImageId.values)  # don't need the dict keys
            
            if not subset_files:
                subset_files = files
                
            # Add images
            self.sub_annotations = annotations[annotations.ImageId.isin(subset_files)]
            for row in self.sub_annotations.drop_duplicates(subset='ImageId').iterrows():
                image_path = os.path.join(dataset_dir, row[1]['ImageId'])
                if image_size=='load':
                    image = PIL.Image.open(image_path)
                    height, width = image.size[::-1]
                else:
                    height, width=image_size[0],image_size[1]
                
                self.add_image("ship",image_id=row[1]['ImageId'], path=image_path,width=width,height=height)
#                 if str(row[1]['EncodedPixels'])!='nan':
#                     self.add_image("ship",image_id=row[1]['ImageId'], path=image_path,width=width,height=height)
#                 else:
#                     self.add_image("ship",image_id=row[1]['ImageId'], path=image_path,width=width,height=height)
    
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        height, width = 768,768
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]

        info = self.image_info[image_id]
        file = info['id']
        df = self.sub_annotations[self.sub_annotations.ImageId==file]

        mask = np.zeros([info['height'], info['width'],len(df)])
        for i, row in enumerate(df.iterrows()):
            if str(row[1]['EncodedPixels'])!='nan':
                mask[:,:,i]= rleToMask(row[1]['EncodedPixels'],info['height'], info['width'])

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)#.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

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
            
class Detection:
    def __init__(self, task):
        self.task = task
        with open('config.json') as f:
            self.conf = json.load(f)
    
    def load_data(self, data_type, file_list=''):
        '''
        #data_type: coco/via/rle
        files_list - can be provided, if subset of files is desired
        '''
        
        dataset_dirs = [self.conf['dataset_dir']+'train/']
        files=[]  
        
        if not file_list:
            print('Starting data parsing from folder')
            for dir in dataset_dirs: # the glob also slows things
                files+=glob.glob(dir+'/*.*')
        else:
            print('Starting data parsing list')
            files=file_list
            
        #full_train_files, val_files = train_test_split(files,random_state=1)
        files = [file.split("/")[-1] for file in files if 'csv' not in file and 'json' not in file]
        train_files, val_files = train_test_split(files,random_state=1)
        print(f'Loading {self.task} data')
        print(data_type)
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
            
        elif data_type=='rle':
            self.val_dataset = RLEDataset()
            self.val_dataset.load_images(dataset_dirs, val_files, 'val')
            self.val_dataset.prepare()
            print(f'finished loading val {len(val_files)}, now loading train')
            self.train_dataset = RLEDataset()
            self.train_dataset.load_images(dataset_dirs, train_files)
            self.train_dataset.prepare()
            # Must call before using the dataset

        self.test_dataset = self.val_dataset
        print("Images: {}\nClasses: {}".format(len(self.train_dataset.image_ids), self.train_dataset.class_names))
        print("Images: {}\nClasses: {}".format(len(self.val_dataset.image_ids), self.val_dataset.class_names))

    def load_model(self, init_with, mode = 'training'):
        #init with: last/coco
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        
        if mode=='inference':
            print('inference mode')
            self.config.NAME = self.task
            self.model = modellib.MaskRCNN(mode="inference", config=self.config, model_dir=MODEL_DIR)

        elif mode == 'training':
            print('training mode')
            self.config.NAME = self.task
            self.model = modellib.MaskRCNN(mode="training", config=self.config, model_dir=MODEL_DIR)

        if init_with == "imagenet":
            self.model.load_weights(model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            COCO_MODEL_PATH = os.path.join(self.conf['model_path'], "mask_rcnn_coco.h5")
            self.model.load_weights(COCO_MODEL_PATH, by_name=True,exclude=["mrcnn_class_logits","mrcnn_bbox_fc","mrcnn_bbox","mrcnn_mask"])
        elif init_with == "last":
            # Load the last model you trained and continue training
            print(f'load model from {self.model.find_last()}')
            self.model.load_weights(self.model.find_last(), by_name=True) #[1]
        return self.model

    def evaluate(self, n=10):
        # main eval function
        # evaluate mAP for crop.
        # version can be '' or banana
        image_ids = self.test_dataset.image_ids# np.random.choice(dataset.image_ids, 10)

        APs = []
        for image_id in image_ids[:n]:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(self.test_dataset, self.config, image_id, use_mini_mask=False)
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
                print('error in copmute')
            APs.append(AP)

        print("mAP: ", np.mean(APs))
        return AP, APs

    def load_data_model_results(self,config, mode = 'inference',data_type='rle',weights='coco', metric ='map'):
        '''
        mode: 'train', inference
        weights: 'coco','last'
        '''
        aug = get_aug()
        self.config=config
        self.load_data(data_type)
        if mode == 'inference':
            self.model = self.load_model(weights, mode)
        if mode == 'training':
            self.model = self.load_model(weights, mode)
            self.model.train(self.train_dataset, self.val_dataset, learning_rate=config.LEARNING_RATE, epochs=30,  layers='heads',
                         augmentation = aug)
        # calc metric
        if metric=='map':
            self.evaluate()
     