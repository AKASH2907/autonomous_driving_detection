import cv2
import numpy as np
import os
from os.path import isfile, join
from os import rename, listdir, rename, makedirs
from matplotlib import pyplot as plt
from shutil import copyfile, move
import fnmatch
import glob
from PIL import Image
import skimage.io as io
import statistics as stat

ann = "./idd20k_lite/gtFine/val/"
nann = "./idd_20k/predictions/res50_psp_10_384/test/test_pixel/"
source = "./idd_20k/predictions/res50_psp_10_384/test/test_final/"

test = "./idd20k_lite/imgs/test/"
files = listdir(test)


# for i in files:
# 	if not os.path.exists(i):
# 		os.makedirs(join(source, i))

def val_evaluation():
	gts = []
	files.sort()
	for i in files[:]:

		g = glob.glob(ann + i + '/*_label.png')
		g = [ lab for lab in g if not lab.endswith("_inst_label.png") ]
		# for j in g:
		dest_folder = source + i
		# g0 = g[0]
		for j in range(len(g)):
			g0 = g[j]
			img_file = g0.split(os.path.sep)[-1]
			copyfile(join(nann, img_file), join(dest_folder, img_file))
		gts+=g
	# print(gts)


def test_submission():
	gts = []
	ground = []
	files.sort()

	for i in files:
		g = glob.glob(test + i + '/*_image.jpg')
		print(g)
		gts+=g
		# folder = g0.split(os.path.sep)[-2]
		folder = source+i
		for j in range(len(g)):
			g0 = g[j]
			img_file = g0.split(os.path.sep)[-1]
			
			ids = img_file[:-10]
			ann_file = ids + "_label.png"
			# ground+=[img_file]
			# copyfile(join(test, i, img_file), join(test_imgs, img_file))
			copyfile(join(nann, ann_file), join(folder, ann_file))

# val_evaluation()
test_submission()
