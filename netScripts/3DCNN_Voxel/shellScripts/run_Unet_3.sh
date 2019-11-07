# [1,2,3] [1,3,5] [2,3,7]-->[123][135][135]
python train_3DCNN_Unet.py --clsname Table --max_epoch 60 --learning_rate 0.001 --gpu 2
python test_3DCNN_Unet.py --clsname Table --withEmpty --gpu 1

