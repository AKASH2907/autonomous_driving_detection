# idd_driving_segmentation
Indian Driving Dataset Segmentation Challenge NCVPRIPG'19

No of submissions left -- 15

# Things to work upon 
Look for repo containing cityscapes dataset and run the codes anyhow 
Cityscapes have void classes look for it

Change the intensity of particular values
Change the name to label.png
Directories make
Test folder some images are not present -> 5 images ID find out


1) divamgupta github repo -> Tomorrow 1st submission
2) qubvel segmentation models
3) Pixel Intensity problem --> Serious

1) UNet

# Concepts to look upon
1) FCN-8, 16, 32 difference
2) 

# Paper to look upon
1) Dealing with 
2) Efficient Net
3) PSP Net
4) 


# Problems Currently facing
1) How to deal with unlabeled pixels in image segmentation?
2) Dealing with instance labels in the dataset
3) 


Questions to ask:
1) In segmentations, what is the image shape of masks? How do you read using cv2 or imageio? Because bothare taking different shapes?
With cv2 it reads 3,3,3 and imageio reads 3
2) Is there any .txt file needed in segmentation like for class information or something? Isn't they learn directly from pixelwise cross
entropy loss?
3) Gud resources for segmentation github that I can use as reference?
4) 

1st set of results
Validation:
---------------------------------------------
Level 1 Scores
---------------------------------------------
drivable		:0.7845120106595769
non-drivable		:0.23014790726948495
living-things		:0.3588385523008605
vehicles		:0.6218017694882831
road-side-objs		:0.34466576972739665
far-objects		:0.6662080565682629
sky		:0.8938274671391853
---------------------------------------------
mIoU		:0.4875001916441313
---------------------------------------------
[0.78466189 0.23014791 0.35885548 0.6218409  0.34473394 0.66637605 0.89399533]
mIoU:				55.723021395073424

