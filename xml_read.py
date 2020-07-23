# split into train and test set
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from os import listdir
from xml.etree import ElementTree



# extract bounding boxes from an annotation file
def extract_boxes(filename):
	# load and parse the file
	tree = ElementTree.parse(filename)
	# get the root of the document
	root = tree.getroot()
	# extract each bounding box
	names = root.findall('.//name')
	for i in names:
		print(i.text)
	boxes = list()

	car_boxes = list()

	for box in root.findall('.//bndbox'):
		xmin = int(box.find('xmin').text)
		ymin = int(box.find('ymin').text)
		xmax = int(box.find('xmax').text)
		ymax = int(box.find('ymax').text)
		coors = [xmin, ymin, xmax, ymax]
		boxes.append(coors)

	c = 0
	
	for i, j in zip(root.findall('.//name'), root.findall('.//bndbox')):
		if i.text=="car":
			car_boxes.append(boxes[c])

		c+=1


	# # extract image dimensions
	# width = int(root.find('.//size/width').text)
	# height = int(root.find('.//size/height').text)
	return boxes, car_boxes

b, c = extract_boxes("1.xml")
print(b, c)



# load the masks for an image
def load_mask(image_id):
	# get details of image
	info = image_info[image_id]
	# define box file location
	path = info['annotation']
	# load XML
	boxes, w, h = extract_boxes(path)
	# create one array for all masks, each on a different channel
	masks = zeros([h, w, len(boxes)], dtype='uint8')
	# create masks
	class_ids = list()
	for i in range(len(boxes)):
		box = boxes[i]
		row_s, row_e = box[1], box[3]
		col_s, col_e = box[0], box[2]
		masks[row_s:row_e, col_s:col_e, i] = 1
		# class_ids.append(self.class_names.index('car'))
	# return masks, asarray(class_ids, dtype='int32')
	# class_ids = np.array([self.class_names.index(s[0]) for s in dataset])
	# for i in range(len(boxes)):
	# 	box = boxes[i]
	# 	row_s, row_e = box[1], box[3]
	# 	col_s, col_e = box[0], box[2]
	# 	masks[row_s:row_e, col_s:col_e, i] = 1
	# 	class_ids.append(self.class_names.index('rider'))			
	return masks, asarray(class_ids, dtype='int32')

	# def load_mask(self, image_id):
	# 	"""Generate instance masks for shapes of the given image ID.
	# 	"""
	# 	info = self.image_info[image_id]
	# 	shapes = info['shapes']
	# 	count = len(shapes)
	# 	mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
	# 	for i, (shape, _, dims) in enumerate(info['shapes']):
	# 		mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
	# 											shape, dims, 1)
	# 	# Handle occlusions
	# 	occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
	# 	for i in range(count-2, -1, -1):
	# 		mask[:, :, i] = mask[:, :, i] * occlusion
	# 		occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
	# 	# Map class names to class IDs.
	# 	class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
	# 	return mask.astype(np.bool), class_ids.astype(np.int32)


