# [1,2,3] [1,3,5] [2,3,7]-->[123][135][135]
python train_3DCNN_Unet.py --clsname Motorbike --max_epoch 150 --learning_rate 0.001 --gpu 1
python test_3DCNN_Unet.py --clsname Motorbike --withEmpty --gpu 1

python train_3DCNN_Unet.py --clsname Earphone --max_epoch 150 --learning_rate 0.001 --gpu 1
python test_3DCNN_Unet.py --clsname Earphone --withEmpty --gpu 1

python train_3DCNN_Unet.py --clsname Rocket --max_epoch 150 --learning_rate 0.001 --gpu 1
python test_3DCNN_Unet.py --clsname Rocket --withEmpty --gpu 1

python train_3DCNN_Unet.py --clsname Bag --max_epoch 150 --learning_rate 0.001 --gpu 1
python test_3DCNN_Unet.py --clsname Bag --withEmpty --gpu 1

python train_3DCNN_Unet.py --clsname Cap --max_epoch 150 --learning_rate 0.001 --gpu 1
python test_3DCNN_Unet.py --clsname Cap --withEmpty --gpu 1

python train_3DCNN_Unet.py --clsname Mug --max_epoch 150 --learning_rate 0.001 --gpu 1
python test_3DCNN_Unet.py --clsname Mug --withEmpty --gpu 1

python train_3DCNN_Unet.py --clsname Skateboard --max_epoch 150 --learning_rate 0.001 --gpu 1
python test_3DCNN_Unet.py --clsname Skateboard --withEmpty --gpu 1

python train_3DCNN_Unet.py --clsname Pistol --max_epoch 150 --learning_rate 0.001 --gpu 1
python test_3DCNN_Unet.py --clsname Pistol --withEmpty --gpu 1

