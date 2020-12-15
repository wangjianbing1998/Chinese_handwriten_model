#nohup python train.py --model vgg16 >output_modelVGG16.txt 2>&1 &
#tail -f output_modelVGG16.txt

#python train.py --model simple --use_test_time
python train.py --model line --use_test_time
