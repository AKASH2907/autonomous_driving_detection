from shutil import copyfile
from os.path import join

f = open("highquality_val.txt", "r")

c = 0
for x in f:
	# print(x)
	copyfile("JPEGImages/" + x[:-1] + ".jpg", join("highquality_test_imgs/" + str(c) + ".jpg"))
	# copyfile("Annotations/" + x[:-1] + ".xml", join("highquality_annots/" + str(c) + ".xml"))
	# break

	if c%100==0:
		print(c)

	c+=1