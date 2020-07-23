import cv2
import numpy as np
from os.path import isfile, join
from os import rename, listdir, rename, makedirs
from matplotlib import pyplot as plt
from shutil import copyfile, move
import fnmatch
import glob
from PIL import Image
import skimage.io as io
import statistics as stat
 
# imgfiles = "./idd20k_lite/imgs" 
# gtMask = "./idd20k_lite/gtFine"


nf_1 = "./idd_20k/semantic"
# nf_2 = "./idd_20k/semantics"


# im_1 = "./idd_20k/imgs"
im_2 = "./idd_20k/nimgs"

distributions = listdir(nf_1)

print(distributions[:1])
for i in distributions[1:]:

	masks = join(nf_1, i)
	imgs = join(im_2, i)
	print(imgs, masks)
	imgfiles = listdir(imgs)
	maskfiles = listdir(masks)
	# print(imgfiles)
	c=0
	for j, k in zip(imgfiles, maskfiles):
		im_file = join(imgs, j)
		msk_file = join(masks, k)

		im = io.imread(im_file)
		msk = io.imread(msk_file)
		# print(msk)
		# print(msk[225])
		# print(stat.mode(msk[225]))
		modes = stat.mode(msk[225])
		# print(type(modes))
		# print(msk[226])
		
		
		uni = list(np.unique(msk))
		c+=1
		if max(uni)>6:
			print(j, k)
			# print(uni)
			for x in range(msk.shape[0]):

				for y in range(msk.shape[1]):
					if msk[x][y]>6:
						# print(msk[x])
						# print(msk[x][y])
						lst = list(msk[x])
						lst_count = [z for z in set(lst)]
						count = [lst.count(z) for z in set(lst)]
						g = np.argmax(count)
						# print(g)
						# # for _ in lst_count:
						# # # 	if 
						# print(lst_count)
						# print(count)
						# print(lst_count[g])
						
						msk[x][y] = lst_count[g]
						# print(stat.mode(msk[x]))
						# print(stat.mean(msk[x]))
						# msk[x][y]=stat.mode(msk[x])

			# io.imsave(join(masks, k), msk)
					# if x!=226 and msk[x][y]>6:
					# 	print(msk[x][y])
					# 	print(x, y)
					# 	msk[x][y] = int((msk[x-1][y-1] + msk[x][y-1] + msk[x+1][y-1] + msk[x-1][y] + 
					# 		msk[x+1][y] + msk[x-1][y+1] + msk[x][y+1] + msk[x+1][y+1])/8)
					# 	print(int((msk[x-1][y-1] + msk[x][y-1] + msk[x+1][y-1] + msk[x-1][y] + 
					# 		msk[x+1][y] + msk[x-1][y+1] + msk[x][y+1] + msk[x+1][y+1])/8))
					# if x==226 and msk[x][y]
			# plt.figure()
			# plt.title(j)
			# plt.imshow(im)
			# plt.figure()
			# plt.title(k)
			# plt.imshow(msk)
			# plt.show()
			# break
		# if c==1:
		# 	break
	    	
	# break
		# im = Image.open(file)
		# # print(im)
		# nfiles = j[:-10]
		# nfiles += ".png"
		# # print(nfiles)
		# im.save(join(nimgs, nfiles))
		# img = listdir(file)
		# print(len(img))
		# print(file)
		# nfile = file[:-10]
		
		# nfile = nfile+".png"
		# print(nfile)	
		# copyfile(file, nfile)
		# if img.endswith()
		
		# for k in img:
		# 	# print(k[-14:-4])
		# 	if k[-9:-4:] == "label":
		# 		# print(k)
		# 		copyfile(join(file, k), join(nimgs, k))
		# 	# for k in glob.glob("*_inst_label.png"):
		# 	# 	print(k)
		# 	# if k.endswith('*_label.png'):
		# 	# 	c+=1
		# 	# 	print(k)
		# # 	break	
		# # break
			# copyfile(join(file, k), join(nimgs, k))
			# break
			# im = cv2.imread(join(file, k))
	# 		if im.shape[0]!=227 and im.shape[1]!=320:
	# 			print(False)
			# print(im.shape)
			# break

plt.show()