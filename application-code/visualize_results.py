""" Visualize Filter Responces """
""" Original Author: Haoqiang Fan """
import numpy as np
import show3d_balls
import sys
import os
import pdb
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--clsname', default='')
parser.add_argument('--atrous_block_num', type=int, default=2)
FLAGS = parser.parse_args()

show_dict = {'Airplane': [[1,15],[1,31],
                         [2, 3],[2,19],[2,56],[2,47],
                         [3, 7],[3,23],[3,24],[3,27],[3,50],[3,55]],  # [stage,channel]}
             'Car': [[1,10],[1,11],[1,27],
                    [2,20],[2,30],[2,51],
                    [3, 1],[3,11],[3,12],[3,21],[3,33],[3,46]],
             # 'Motorbike': [[1,26],[1, 1],
             #              [2, 0],[2,19],[2,27],
             #              [3, 0],[3, 2],[3, 4],[3,27],[3,40],[3,42],[3,47],[3,55],[3,46]],
             'Chair': [[1,16],[1,20],
                       [2, 0],[2, 1],[2,23],[2,44],[2,55],
                       [3, 3],[3,25],[3,28],[3,45],[3,37]],
            }

# for atrous 1st layer, 2nd layer, 3rd layer
show_dict_2 = {'Airplane': [#[1,0],[1,21],[1,28],
                         # [2, 0],[2, 1],[2,12],[2,14],[2,31],
                         [3, 2],[3, 9],[3,17],[3,22],[3,19]],  # [stage,channel]}
             'Car': [[1,5],[1, 9],
                    [2,14],[2,29],[2,30],
                    [3, 9],[3,15],[3,27],[3,46],[3,47],[3,54],[3,57]],
             # 'Motorbike': [[1,26],[1, 1],
             #              [2, 0],[2,19],[2,27],
             #              [3, 0],[3, 2],[3, 4],[3,27],[3,40],[3,42],[3,47],[3,55],[3,46]],
             'Chair': [[1,19],[1,27],
                       [2, 3],[2, 4],[2, 9],[2,21],
                       [3, 0],[3,18],[3,26],[3,29],[3,33],[3,35]],
            }

if __name__=='__main__':
    CLSNAME = 'Airplane'
    feature_folder = 'AtrousBlock2_jet'
    # feature_folder = 'AtrousBlock2-Encode_jet'

    if feature_folder == 'AtrousBlock2_jet':
        test_dir = os.path.join(BASE_DIR, 'AtrousBlock2_jet', CLSNAME+'-withBG-ABlock2-Res')
        show_list = show_dict[CLSNAME]  # [stage,channel]
    elif feature_folder == 'AtrousBlock2-Encode_jet':
        test_dir = os.path.join(BASE_DIR, 'AtrousBlock2-Encode_jet', CLSNAME+'-withBG-ABlock2-Res')
        show_list = show_dict_2[CLSNAME]  # [stage,channel]

    for shape in range(5):
        for stch in show_list:
            stage = stch[0]
            channel = stch[1]
            fname = os.path.join(test_dir, 'shape_'+str(shape)+'-stage_'+str(stage)+'-ch_'+str(channel)+'.npy')
            ptsclr = np.load(fname)  # shape (n,7)
            clr = ptsclr[:,3:6] * 255
            # pdb.set_trace()
            savedir = os.path.join(test_dir, 'dump', 'shape_'+str(shape)+'-stage_'+str(stage)+'-ch_'+str(channel)+'.png')
            show3d_balls.showpoints(xyz=ptsclr[:,:3], c_pred=clr[:,[1,0,2]], background=(255,255,255),
                showrot=False,magnifyBlue=0,freezerot=False,
                normalizecolor=False, ballradius=12, savedir=savedir)
