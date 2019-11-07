# [1,2,3] [1,3,5] [2,3,7]-->[123][135][135]
#python3 train_3DCNN_Atrous-coseg.py --atrous_block_num 3 --clsname cosegTeleAliens --max_epoch 300 --learning_rate 0.001 --gpu 0
python3 test_3DCNN_Atrous-coseg.py --atrous_block_num 3 --clsname cosegTeleAliens --withEmpty --gpu 0

#python3 train_3DCNN_Atrous-coseg.py --atrous_block_num 3 --clsname cosegChairsLarge --max_epoch 300 --learning_rate 0.001 --gpu 0
python3 test_3DCNN_Atrous-coseg.py --atrous_block_num 3 --clsname cosegChairsLarge --withEmpty --gpu 0

#python3 train_3DCNN_Atrous-coseg.py --atrous_block_num 3 --clsname cosegVasesLarge --max_epoch 300 --learning_rate 0.001 --gpu 0
python3 test_3DCNN_Atrous-coseg.py --atrous_block_num 3 --clsname cosegVasesLarge --withEmpty --gpu 0

