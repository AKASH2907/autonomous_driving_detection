

python train_rpn.py --network vgg -o simple -p bboxes.txt

python train_frcnn.py --network vgg -o simple -p bboxes.txt --rpn models/rpn/rpn.vgg.weights.04-2.59.hdf5


python test_frcnn.py -p ../test_images/ --load models/vgg/voc.hdf5 
