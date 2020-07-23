# split into train and test set
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from matplotlib import pyplot
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
import numpy as np
from numpy import zeros
from numpy import asarray
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.models import load_model
from os import listdir
from xml.etree import ElementTree
import matplotlib.pyplot as plt


class autonomousConfig(Config):
	# give the configuration a recognizable name
	NAME = "autonomous"
 
	# set the number of GPUs to use along with the number of images
	# per GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
 
	# number of classes (we would normally add +1 for the background)
	 # kangaroo + BG
	NUM_CLASSES = 1+2
   
	# Number of training steps per epoch
	STEPS_PER_EPOCH = 131
	
	# Learning rate
	LEARNING_RATE=0.006
	
	# Skip detections with < 90% confidence
	DETECTION_MIN_CONFIDENCE = 0.9
	
	# setting Max ground truth instances
	# MAX_GT_INSTANCES=10


# class that defines and loads the kangaroo dataset
class autonomousDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "car")
		self.add_class("dataset", 2, "rider")
		# define data locations
		images_dir = dataset_dir + '/images_mod/'
		annotations_dir = dataset_dir + '/annots_mod/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			# skip bad images
			# if image_id in ['00090']:
			# 	continue
			# skip all images after 150 if we are building the train set
			if is_train and int(image_id) >= 3000:
				continue
			# skip all images before 150 if we are building the test/val set
			if not is_train and int(image_id) < 3000:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids=[0,1,2])

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# c= 0
		# extract each bounding box
		boxes = list()
		car_boxes = list()

		for box in root.findall('.//object'):
			name = box.find('name').text
			xmin = int(box.find('./bndbox/xmin').text)
			ymin = int(box.find('./bndbox/ymin').text)
			xmax = int(box.find('./bndbox/xmax').text)
			ymax = int(box.find('./bndbox/ymax').text)
			coors = [xmin, ymin, xmax, ymax, name]
			if name=='car' or name=='rider':
				boxes.append(coors)
		# for i, j in zip(root.findall('.//name'), root.findall('.//bndbox')):
		# 	if i.text=="car":
		# 		car_boxes.append(boxes[c])

		# 	c+=1

		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# print(boxes)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			# print(box)
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			if (box[4] == 'car'):
				masks[row_s:row_e, col_s:col_e, i] = 1
				class_ids.append(self.class_names.index('car'))
			else:
				masks[row_s:row_e, col_s:col_e, i] = 2
				class_ids.append(self.class_names.index('rider'))
		# return masks, asarray(class_ids, dtype='int32')
		# class_ids = np.array([self.class_names.index(s[0]) for s in dataset])
		# for i in range(len(boxes)):
		# 	box = boxes[i]
		# 	row_s, row_e = box[1], box[3]
		# 	col_s, col_e = box[0], box[2]
		# 	masks[row_s:row_e, col_s:col_e, i] = 1
		# 	class_ids.append(self.class_names.index('rider'))			
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']


config = autonomousConfig()
# config.display()

# train set
train_set = autonomousDataset()
train_set.load_dataset('autonomous', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# prepare test/val set
test_set = autonomousDataset()
test_set.load_dataset('autonomous', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
br
# # define the model
# model = MaskRCNN(mode='training', model_dir='./', config=config)
# # # load weights (mscoco) and exclude the output layers
# model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# # # train weights (output layers or 'heads')
# model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')

# model_path = 'multi_objects.h5'
# model.keras_model.save_weights(model_path)


# load an image
image_id = 3
image = train_set.load_image(image_id)
# load the masks and the class ids
mask, class_ids = train_set.load_mask(image_id)
# extract bounding boxes from the masks
bbox = extract_bboxes(mask)
# display image with masks and bounding boxes
img = display_instances(image, bbox, mask, class_ids, train_set.class_names)
plt.savefig('masked_1.png', bbox_inches='tight', pad_inches=0)

# for i in range(10):
# 	image_id = i

# 	image = train_set.load_image(image_id)
# 	# load the masks and the class ids
# 	mask, class_ids = train_set.load_mask(image_id)
# 	# extract bounding boxes from the masks
# 	bbox = extract_bboxes(mask)
# 	# display image with masks and bounding boxes
# 	img = display_instances(image, bbox, mask, class_ids, train_set.class_names)
# 	plt.savefig('mask_samples/masked_' + str(i) + '.png', bbox_inches='tight', pad_inches=0)


# print(image.shape)
# # load image mask
# mask, class_ids = train_set.load_mask(image_id)
# print(mask.shape)
# # plot image
# pyplot.imshow(image)
# # plot mask
# pyplot.imshow(mask[:, :, 0], cmap='gray', alpha=0.5)
# pyplot.show()