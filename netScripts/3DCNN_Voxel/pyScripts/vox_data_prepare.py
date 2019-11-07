""" Voxel data prepare functions

Author: WZJ
Date: April 2017
"""
import os
import sys
import numpy as np
import h5py
import pdb

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
# print("BASE_DIR = " + BASE_DIR)
# print(sys.path)
import pc_util
import data_prep_util
vox_size = 64  # 48,32

data_Flag = 3  # {1, 2} : 1 for generating vox wise labeling data; 2 for boundary detection; 3 for Vox CNN feature extraction net


if(data_Flag == 2):  # for boundary detection
    # test data ----------------------------------------
    for fn in range(2):
        h5filename = os.path.join(BASE_DIR, '3DCNN', 'hdf5_data', 'vol_data_test'+str(fn)+'.h5')
        [data, label, seg] = data_prep_util.load_h5_data_label_seg_vol(h5filename)
        print(h5filename)
        vol_bon_list = pc_util.part_boundary_detection_batch(vol_segs=seg)
        h5_out_name = os.path.join(BASE_DIR, '3DCNN', 'hdf5_data', 'bon_data_test'+str(fn)+'.h5')
        data_prep_util.save_h5_bon(h5_out_name, data=data, label=label, boundary=vol_bon_list)
    # val data ----------------------------------------
    for fn in range(1):
        h5filename = os.path.join(BASE_DIR, '3DCNN', 'hdf5_data', 'vol_data_val'+str(fn)+'.h5')
        [data, label, seg] = data_prep_util.load_h5_data_label_seg_vol(h5filename)
        print(h5filename)
        vol_bon_list = pc_util.part_boundary_detection_batch(vol_segs=seg)
        h5_out_name = os.path.join(BASE_DIR, '3DCNN', 'hdf5_data', 'bon_data_val'+str(fn)+'.h5')
        data_prep_util.save_h5_bon(h5_out_name, data=data, label=label, boundary=vol_bon_list)
    # train data ----------------------------------------
    for fn in range(6):
        h5filename = os.path.join(BASE_DIR, '3DCNN', 'hdf5_data', 'vol_data_train'+str(fn)+'.h5')
        [data, label, seg] = data_prep_util.load_h5_data_label_seg_vol(h5filename)
        print(h5filename)
        vol_bon_list = pc_util.part_boundary_detection_batch(vol_segs=seg)
        h5_out_name = os.path.join(BASE_DIR, '3DCNN', 'hdf5_data', 'bon_data_train'+str(fn)+'.h5')
        data_prep_util.save_h5_bon(h5_out_name, data=data, label=label, boundary=vol_bon_list)


if(data_Flag == 1):  # for generating vox wise labeling data
    # test data ----------------------------------------
    for fn in range(2):
        h5filename = os.path.join(BASE_DIR, 'part_seg', 'hdf5_data', 'ply_data_test'+str(fn)+'.h5')
        [data, label, seg] = data_prep_util.load_h5_data_label_seg(h5filename)
        print(h5filename)
        [vol_list, pid_vol_list] = pc_util.point_cloud_to_volume_batch_wzj(point_clouds=data, part_labels=seg, vsize=vox_size)
        h5_out_name = os.path.join(BASE_DIR, '3DCNN', 'hdf5_data', str(vox_size), 'vol_data_test'+str(fn)+'.h5')
        data_prep_util.save_h5_vol(h5_out_name, data=vol_list, label=label, seg=pid_vol_list)

    # validation data ----------------------------------------
    for fn in range(1):
        h5filename = os.path.join(BASE_DIR, 'part_seg', 'hdf5_data', 'ply_data_val'+str(fn)+'.h5')
        [data, label, seg] = data_prep_util.load_h5_data_label_seg(h5filename)
        print(h5filename)
        [vol_list, pid_vol_list] = pc_util.point_cloud_to_volume_batch_wzj(point_clouds=data, part_labels=seg, vsize=vox_size)
        h5_out_name = os.path.join(BASE_DIR, '3DCNN', 'hdf5_data', str(vox_size), 'vol_data_val'+str(fn)+'.h5')
        data_prep_util.save_h5_vol(h5_out_name, data=vol_list, label=label, seg=pid_vol_list)

    # train data ----------------------------------------
    for fn in range(6):
        h5filename = os.path.join(BASE_DIR, 'part_seg', 'hdf5_data', 'ply_data_train'+str(fn)+'.h5')
        [data, label, seg] = data_prep_util.load_h5_data_label_seg(h5filename)
        print(h5filename)
        [vol_list, pid_vol_list] = pc_util.point_cloud_to_volume_batch_wzj(point_clouds=data, part_labels=seg, vsize=vox_size)
        h5_out_name = os.path.join(BASE_DIR, '3DCNN', 'hdf5_data', str(vox_size), 'vol_data_train'+str(fn)+'.h5')
        data_prep_util.save_h5_vol(h5_out_name, data=vol_list, label=label, seg=pid_vol_list)


if(data_Flag == 3):  # [20170920] for generating data, Vox CNN feature extraction net
    File_Batch_Size = 200
    # define some util funcs

    def rand_sample_points(pts, plb, num=20000):  # 15000
        points_num = pts.shape[0]
        idx = np.random.permutation(num)
        if points_num < num:
            idx = np.mod(idx, points_num)
        pts_new = pts[idx, ...]
        plb_new = plb[idx, ...]
        return pts_new, plb_new

    def save_h5_points_data(h5_filename, data, label, ptsnum,
                            data_dtype='float32', label_dtype='uint8'):
        h5_fout = h5py.File(h5_filename, 'w')
        h5_fout.create_dataset(
                'data', data=data,
                compression='gzip', compression_opts=4,
                dtype=data_dtype)
        h5_fout.create_dataset(
                'label', data=label,
                compression='gzip', compression_opts=1,
                dtype=label_dtype)
        h5_fout.create_dataset(
                'ptsnum', data=ptsnum,
                compression='gzip', compression_opts=1,
                dtype='int32')
        h5_fout.close()

    def save_h5_volumes_data(h5_filename, data, seg, data_dtype='uint8', seg_dtype='uint8'):
        h5_fout = h5py.File(h5_filename, 'w')
        h5_fout.create_dataset(
                'data', data=data,
                compression='gzip', compression_opts=4,
                dtype=data_dtype)
        h5_fout.create_dataset(
                'seg', data=seg,
                compression='gzip', compression_opts=1,
                dtype=seg_dtype)
        h5_fout.close()

    # get file list
    ROOT_DIR = '/home/wzj/3D_Shape_proj/Data'
    split_root = os.path.join(ROOT_DIR, 'train_test_split/')
    # for Ocnet data, generate new h5 files
    category_list = ['Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife',
           'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table']  # ['Motorbike', 'Earphone', 'Rocket', 'Airplane', 'Chair']
    # category_list = ['Table']  # ['Motorbike', 'Earphone', 'Rocket', 'Airplane', 'Chair']
    # class_name = 'Motorbike'  # 'Motorbike', 'Earphone', 'Rocket', 'Airplane', ['Chair']
    for class_name in category_list:
        # for train files
        print(class_name)
        for train_or_test in ['train', 'test', 'val']:
            file_list = split_root + class_name + '_' + train_or_test + '_file_list.txt'
            file_names = []
            with open(file_list) as f:
                for line in f.readlines():
                    file_names.append(line.rstrip('\n'))
            """ if there are too many files, process in batch """
            pts_h5dir = os.path.join(ROOT_DIR, 'ocnet', 'PointsData', 'hdf5_data', str(vox_size), class_name+'_pts_'+train_or_test+'.h5')
            vol_h5dir = os.path.join(ROOT_DIR, 'ocnet', 'PointsData', 'hdf5_data', str(vox_size), class_name+'_vol_'+train_or_test+'.h5')
            num_file = len(file_names)
            if num_file > File_Batch_Size:
                pts_h5_fout = h5py.File(pts_h5dir, 'a')
                vol_h5_fout = h5py.File(vol_h5dir, 'a')
                num_batches = np.ceil(num_file / File_Batch_Size).astype(int)
                for batch_idx in range(num_batches):
                    print('batch id: ', batch_idx+1, ' / ', num_batches)
                    start_idx = batch_idx * File_Batch_Size
                    end_idx = (batch_idx + 1) * File_Batch_Size
                    if min(num_file-start_idx, end_idx-start_idx) < File_Batch_Size:
                        batch_files = file_names[start_idx:min(end_idx, num_file)]
                    else:
                        batch_files = file_names[start_idx:end_idx]
                    points = []
                    labels = []
                    point_num = []
                    i = 0
                    for file in batch_files:
                        i = i+1
                        print(train_or_test, ' set: ', str(i) + '/' + str(len(batch_files)) + ' ' + file)
                        pts_file = os.path.join(ROOT_DIR, 'ocnet', 'PointsData', file+'.pts.txt')
                        plb_file = os.path.join(ROOT_DIR, 'ocnet', 'PointsData', file+'.plb.txt')
                        try:
                            pts = np.loadtxt(pts_file)
                            plb = np.loadtxt(plb_file)
                            # pts_sam, plb_sam = rand_sample_points(pts, plb)  # rand sampled points and their labels
                            # points.append(pts_sam)
                            # labels.append(plb_sam)
                            points.extend(pts)
                            labels.extend(plb)
                            point_num.append(pts.shape[0])
                        except FileNotFoundError as e:
                            print('Error! ', e)
                            print('no file:', file)
                            continue
                    data = np.stack(points, axis=0)  # data has shape (total_ptsnum, 3) for Motor train
                    label = np.stack(labels, axis=0)
                    ptsnum = np.asarray(point_num, dtype=np.int32)  # default int64
                    print('point num:\n', point_num)
                    # # SAVE pts h5 files--change to batch saving mode
                    if batch_idx == 0:
                        # pts data
                        pts_data = pts_h5_fout.create_dataset(
                                'data', (data.shape[0], 3), maxshape=(None, 3),
                                compression='gzip', compression_opts=4,
                                dtype='float32')
                        pts_label = pts_h5_fout.create_dataset(
                                'label', (label.shape[0],), maxshape=(None,),
                                compression='gzip', compression_opts=1,
                                dtype='uint8')
                        pts_num = pts_h5_fout.create_dataset(
                                'ptsnum', (ptsnum.shape[0],), maxshape=(None,),
                                compression='gzip', compression_opts=1,
                                dtype='int32')
                        pts_data[...] = data
                        pts_label[:] = label
                        pts_num[:] = ptsnum
                        print('points hdf5 saved.\n\tProcessing vox h5 data')
                        # vol data
                        [vol_stack, pid_vol_stack] = pc_util.point_cloud_to_volume_batch_wzj_v2(point_clouds=data,
                                part_labels=label, ptsnum=ptsnum, vsize=vox_size)

                        vol_data = vol_h5_fout.create_dataset(
                                'data', (vol_stack.shape[0], vox_size, vox_size, vox_size, 1),
                                maxshape=(None, vox_size, vox_size, vox_size, 1),
                                compression='gzip', compression_opts=4,
                                dtype='uint8')
                        vol_seg = vol_h5_fout.create_dataset(
                                'seg', (pid_vol_stack.shape[0], vox_size, vox_size, vox_size, 1),
                                maxshape=(None, vox_size, vox_size, vox_size, 1),
                                compression='gzip', compression_opts=1,
                                dtype='uint8')
                        vol_data[:] = vol_stack
                        vol_seg[:] = pid_vol_stack
                        print('voxel hdf5 saved.')
                    else:
                        # pts data
                        pts_data = pts_h5_fout['data']
                        pts_label = pts_h5_fout['label']
                        pts_num = pts_h5_fout['ptsnum']

                        pts_data.resize(pts_data.shape[0]+data.shape[0], axis=0)
                        pts_label.resize(pts_label.shape[0]+label.shape[0], axis=0)
                        pts_num.resize(pts_num.shape[0]+ptsnum.shape[0], axis=0)

                        pts_data[-data.shape[0]:, ...] = data
                        pts_label[-label.shape[0]:] = label
                        pts_num[-ptsnum.shape[0]:] = ptsnum
                        print('points hdf5 saved.\n\tProcessing vox h5 data')
                        # pdb.set_trace()
                        # vol data
                        [vol_stack, pid_vol_stack] = pc_util.point_cloud_to_volume_batch_wzj_v2(point_clouds=data,
                                part_labels=label, ptsnum=ptsnum, vsize=vox_size)
                        vol_data = vol_h5_fout['data']
                        vol_seg = vol_h5_fout['seg']
                        vol_data.resize(vol_data.shape[0]+vol_stack.shape[0], axis=0)
                        vol_seg.resize(vol_seg.shape[0]+pid_vol_stack.shape[0], axis=0)
                        vol_data[-vol_stack.shape[0]:, ...] = vol_stack
                        vol_seg[-pid_vol_stack.shape[0]:, ...] = pid_vol_stack
                        print('voxel hdf5 saved.')

                    pts_h5_fout.flush()
                    vol_h5_fout.flush()
                pts_h5_fout.close()
                vol_h5_fout.close()
            else:  # num_file <= 800
                points = []
                labels = []
                point_num = []
                i = 0
                for file in file_names:
                    i = i+1
                    print(train_or_test, ' set: ', str(i) + '/' + str(len(file_names)) + ' ' + file)
                    pts_file = os.path.join(ROOT_DIR, 'ocnet', 'PointsData', file+'.pts.txt')
                    plb_file = os.path.join(ROOT_DIR, 'ocnet', 'PointsData', file+'.plb.txt')
                    try:
                        pts = np.loadtxt(pts_file)
                        plb = np.loadtxt(plb_file)
                        # pts_sam, plb_sam = rand_sample_points(pts, plb)  # rand sampled points and their labels
                        # points.append(pts_sam)
                        # labels.append(plb_sam)
                        points.extend(pts)
                        labels.extend(plb)
                        point_num.append(pts.shape[0])
                    except FileNotFoundError as e:
                        print('Error! ', e)
                        continue
                data = np.stack(points, axis=0)  # data has shape (total_ptsnum, 3) for Motor train
                label = np.stack(labels, axis=0)
                ptsnum = np.asarray(point_num, dtype=np.int32)  # default int64
                print('point num:\n', point_num)
                # # SAVE pts h5 files
                # pts_h5dir = os.path.join(ROOT_DIR, 'ocnet', 'PointsData', 'hdf5_data', class_name+'_pts_'+train_or_test+'.h5')
                save_h5_points_data(pts_h5dir, data=data, label=label, ptsnum=ptsnum)
                print('points hdf5 saved.\n\tProcessing vox h5 data')
                # # process voxelization and create h5 data for that
                # pdb.set_trace()
                [vol_stack, pid_vol_stack] = pc_util.point_cloud_to_volume_batch_wzj_v2(point_clouds=data,
                        part_labels=label, ptsnum=ptsnum, vsize=vox_size)
                # vol_bon_stack = pc_util.part_boundary_detection_batch(vol_segs=pid_vol_stack)  # boundary detection
                # # TODO: refine vol_boundary, thinner?
                # # # SAVE vol h5 files
                # vol_h5dir = os.path.join(ROOT_DIR, 'ocnet', 'PointsData', 'hdf5_data', class_name+'_vol_'+train_or_test+'.h5')
                save_h5_volumes_data(vol_h5dir, data=vol_stack, seg=pid_vol_stack)
                print('voxel hdf5 saved.')
            Target_Dir = '/home/wzj/3D_Shape_proj/3DCNN_Voxel/hdf5_data_oc/'+str(vox_size)
            os.system('cp %s %s' % (pts_h5dir, Target_Dir))
            os.system('cp %s %s' % (vol_h5dir, Target_Dir))
