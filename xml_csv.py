import xml.etree.ElementTree as ET
import csv
from os import listdir
import glob
from os.path import join


files = glob.glob(join("highquality_annots", '*.xml'))
files.sort(key=lambda f:int(''.join(filter(str.isdigit, f))))
print(files[:5])
print(len(files))

def coords(filename, im):
	tree = ET.parse(filename)
	root = tree.getroot()
	boxes = list()

	for box in root.findall('.//object'):
		name = box.find('name').text
		xmin = int(box.find('./bndbox/xmin').text)
		ymin = int(box.find('./bndbox/ymin').text)
		xmax = int(box.find('./bndbox/xmax').text)
		ymax = int(box.find('./bndbox/ymax').text)
		coors = [im, xmin, ymin, xmax, ymax, name]
		boxes.append(coors)
	return boxes

classes_list = []
with open('high_bboxes.csv','w') as f:
	writer = csv.writer(f)
	for file in files[:500]:
		data_dir = "../highquality_dataset/images/"
		img_name = file.split(".")[0].split("/")[1] + ".jpg"
		# print(data_dir, img_name)
		# sd
		ordinates = coords(file, data_dir + img_name)
		# print(ordinates)
		# print(len(ordinates))
		for i in ordinates:

			# Optional
			if i[5]=="caravan":
				i[5] = "car"
			if i[5]=="bicycle":
				i[5] = "motorcycle"
			if i[5]=="bus":
				i[5] = "truck"
			if i[5]=="traffic light":
				i[5] = "traffic sign"
			
			#Comment above code according to your requirements
			writer.writerow(i)	
			# classes_list += [i[5]]

f.close()
# classes = set(classes_list)
# print(classes)

# for i in classes:
# 	print(classes_list.count(i))