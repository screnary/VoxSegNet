# [1,2,3] [1,3,5] [2,3,7]-->[123][135][135]
python train_3DCNN_Atrous.py --atrous_block_num 2 --clsname Car --max_epoch 60 --learning_rate 0.001 --gpu 1
python test_3DCNN_Atrous.py --atrous_block_num 2 --clsname Car --withEmpty --gpu 1

python train_3DCNN_Atrous.py --atrous_block_num 2 --clsname Guitar --max_epoch 60 --learning_rate 0.001 --gpu 1
python test_3DCNN_Atrous.py --atrous_block_num 2 --clsname Guitar --withEmpty --gpu 1

python train_3DCNN_Atrous.py --atrous_block_num 2 --clsname Knife --max_epoch 100 --learning_rate 0.001 --gpu 1
python test_3DCNN_Atrous.py --atrous_block_num 2 --clsname Knife --withEmpty --gpu 1

python train_3DCNN_Atrous.py --atrous_block_num 2 --clsname Laptop --max_epoch 60 --learning_rate 0.001 --gpu 1
python test_3DCNN_Atrous.py --atrous_block_num 2 --clsname Laptop --withEmpty --gpu 1

python train_3DCNN_Atrous.py --atrous_block_num 2 --clsname Lamp --max_epoch 60 --learning_rate 0.001 --gpu 1
python test_3DCNN_Atrous.py --atrous_block_num 2 --clsname Lamp --withEmpty --gpu 1

