# [1,2,3] [1,3,5] [2,3,7]-->[123][135][135]
python train_3DCNN_Atrous.py --atrous_block_num 2 --clsname Airplane --max_epoch 60 --learning_rate 0.001 --gpu 2
python test_3DCNN_Atrous.py --atrous_block_num 2 --clsname Airplane --withEmpty --gpu 2

python train_3DCNN_Atrous.py --atrous_block_num 2 --clsname Chair --max_epoch 60 --learning_rate 0.001 --gpu 2
python test_3DCNN_Atrous.py --atrous_block_num 2 --clsname Chair --withEmpty --gpu 2

