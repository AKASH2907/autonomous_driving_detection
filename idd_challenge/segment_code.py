from keras_segmentation.models.unet import vgg_unet, resnet50_unet
from keras_segmentation.models.pspnet import resnet50_pspnet
from keras_segmentation.models.segnet import resnet50_segnet
import keras_segmentation
from keras_segmentation.predict import predict, predict_multiple, evaluate

# model = vgg_segnet(n_classes=7 ,  input_height=320, input_width=320  )
# model = resnet50_segnet(n_classes=7 ,  input_height=320, input_width=320  )
model = resnet50_segnet(n_classes=7 ,  input_height=512, input_width=512)

# model.train(
#     train_images =  "idd_20k/imgs/train/",
#     train_annotations = "idd_20k/semantic/train/",
#     checkpoints_path = "./tmp/resnet50_segnet_1" , epochs=5
# )
# out = model.predict_segmentation(
#     inp="dataset1/images_prepped_test/0016E5_07965.png",
#     out_fname="/tmp/out.png"
# )
# predict_multiple( 
# 	checkpoints_path="./tmp/resnet50_segnet_1", 
# 	inp_dir="./idd_20k/imgs/val/", 
# 	out_dir="./idd_20k/predictions/val/res_seg_5_512/val/" 
# )

# evaluate()