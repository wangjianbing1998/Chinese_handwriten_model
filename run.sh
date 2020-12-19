nohup python train.py --model vgg16 >output_modelVGG16.txt 2>&1 &
nohup python train.py --model vgg19 >output_modelVGG19.txt 2>&1 &
nohup python train.py --model resnet34 >output_modelResnet34.txt 2>&1 &
nohup python train.py --model simple >output_modelSimple.txt 2>&1 &
nohup python train.py --model line >output_modelLine.txt 2>&1 &
#tail -f output_modelVGG16.txt

#python train.py --model simple --use_test_time
#python train.py --model line --use_test_time
