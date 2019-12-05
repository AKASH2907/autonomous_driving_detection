# idd_driving_segmentation
Indian Driving Dataset Segmentation Challenge NCVPRIPG'19

No of submissions left -- 14

# Codes to work upon 
Look for repo containing cityscapes dataset and run the codes anyhow 
Cityscapes have void classes look for it
1) divamgupta github repo -> Try segnet and all rest of the models
2) qubvel segmentation models 
2) Efficient Net
3) PSP Net
4) Deep Lab V3
5) kNN -> Nearest pixel -> Class identification + Dealing with void pixels

# Problems Currently facing
1) How to deal with unlabeled pixels in image segmentation?
2) Dealing with instance labels in the dataset
3) 

# Concepts to look upon
1) FCN-8, 16, 32 difference
2) 

1st set of results
Validation:
---------------------------------------------
Level 1 Scores | mIoU
-------------|--------------------------------
drivable		|0.7845120106595769
non-drivable	|	0.23014790726948495
living-things	|	0.3588385523008605
vehicles		| 0.6218017694882831
road-side-objs	| 0.34466576972739665
far-objects		| 0.6662080565682629
sky		| 0.8938274671391853
**mIoU**		| **0.4875001916441313**


[0.78466189 0.23014791 0.35885548 0.6218409  0.34473394 0.66637605 0.89399533]

mIoU:				55.723021395073424


# Submissions 
S. No. |Architecture | mIoU | Comments
-------|--------------|------|---------
1 | Res50+UNet| 0.404 | 10 epochs + residual pixels==0


# Solved
* Change the intensity of particular values
* Change the name to label.png
* Directories make
* Test folder some images are not present -> 5 images ID find out
