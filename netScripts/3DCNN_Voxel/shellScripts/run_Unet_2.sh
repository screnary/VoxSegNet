# [1,2,3] [1,3,5] [2,3,7]-->[123][135][135]
python train_3DCNN_Unet.py --clsname Car --max_epoch 100 --learning_rate 0.001 --gpu 1
python test_3DCNN_Unet.py --clsname Car --withEmpty --gpu 1

python train_3DCNN_Unet.py --clsname Guitar --max_epoch 100 --learning_rate 0.001 --gpu 1
python test_3DCNN_Unet.py --clsname Guitar --withEmpty --gpu 1

python train_3DCNN_Unet.py --clsname Knife --max_epoch 100 --learning_rate 0.001 --gpu 1
python test_3DCNN_Unet.py --clsname Knife --withEmpty --gpu 1

python train_3DCNN_Unet.py --clsname Laptop --max_epoch 100 --learning_rate 0.001 --gpu 1
python test_3DCNN_Unet.py --clsname Laptop --withEmpty --gpu 1

python train_3DCNN_Unet.py --clsname Airplane --max_epoch 100 --learning_rate 0.001 --gpu 1
python test_3DCNN_Unet.py --clsname Airplane --withEmpty --gpu 1

python train_3DCNN_Unet.py --clsname Lamp --max_epoch 100 --learning_rate 0.001 --gpu 1
python test_3DCNN_Unet.py --clsname Lamp --withEmpty --gpu 1

python train_3DCNN_Unet.py --clsname Chair --max_epoch 60 --learning_rate 0.001 --gpu 1
python test_3DCNN_Unet.py --clsname Chair --withEmpty --gpu 1
