import os
import sys
import numpy as np
import h5py
import pdb
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
# DATA_DIR = os.path.join(BASE_DIR, 'data')
# if not os.path.exists(DATA_DIR):
    # os.mkdir(DATA_DIR)
# if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    # www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    # zipfile = os.path.basename(www)
    # os.system('wget %s; unzip %s' % (www, zipfile))
    # os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    # os.system('rm %s' % (zipfile))


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def point_cloud_to_volume(points, plb, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
        (actually, after nomalization, all points are in a unit sphere)
    """
    vol = np.zeros((vsize,vsize,vsize))
    vseg = np.zeros((vsize,vsize,vsize))
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel  # shift all points to non-negative coordinates, assign to an occupancy grid
    locations = locations.astype(int)
    vol[locations[:,0],locations[:,1],locations[:,2]] = 1.0
    vseg[locations[:,0],locations[:,1],locations[:,2]] = plb
    return vol, vseg


def rotate_voxel_data(batch_data, batch_seg, axis='y'):
    """ rotate voxel data """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    rotated_seg = np.zeros(batch_seg.shape, dtype=np.float32)
    vsize = batch_data.shape[1]
    for k in range(batch_data.shape[0]):
        rot_Flag = True
        if np.random.uniform()<0.6:
            rot_Flag = False
            rot_vox = np.squeeze(batch_data[k,...])
            rot_seg = batch_seg[k,...]
        if rot_Flag:
            rotation_angle = np.random.randint(1,11) * np.pi / 6
            vox = np.squeeze(batch_data[k,...])
            vox_pts = np.transpose(np.array(np.where(vox)))
            seg = batch_seg[k, vox_pts[:,0], vox_pts[:,1], vox_pts[:,2]]
            norm_pts = pc_normalize(vox_pts)

            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            if axis=='y':
                rotation_matrix = np.array([[cosval, 0, sinval],
                                            [0, 1, 0],
                                            [-sinval, 0, cosval]])
            elif axis=='z':
                rotation_matrix = np.array([[cosval, -sinval, 0],
                                            [sinval, cosval, 0],
                                            [0, 0, 1]])
            elif axis=='x':
                rotation_matrix = np.array([[1, 0, 0],
                                            [0, cosval, -sinval],
                                            [0, sinval, cosval]])
            rot_pts = np.dot(norm_pts.reshape((-1, 3)), rotation_matrix)
            rot_vox, rot_seg = point_cloud_to_volume(rot_pts, seg, vsize)

        rotated_data[k,...] = np.expand_dims(rot_vox, axis=-1)
        rotated_seg[k,...] = rot_seg
    return rotated_data, rotated_seg


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def loadDataFile(filename):
    return load_h5(filename)


def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)


def load_h5_data_label_seg_vol(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['seg'][:]
    return (data, label, seg)


def load_h5_volumes_data(h5_filename):
    """ WZJ-20170921 """
    f = h5py.File(h5_filename)
    data = f['data'][:]
    seg = f['seg'][:]
    bon = []
    return (data, seg, bon)


def load_h5_ptsfea_data(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    num = f['num'][:]
    return (data, label, num)


def load_h5_ptsfealoc_data(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    num = f['num'][:]
    loc = f['loc'][:]
    return (data, label, num, loc)


def load_h5_mask(h5_filename):
    f = h5py.File(h5_filename, 'r')
    bbox = f['bbox'][:]
    mask = f['mask'][:]
    label = f['label'][:]
    return (mask, bbox, label)


def load_pts_oc_h5(h5_filename):
    f = h5py.File(h5_filename)
    print([key for key in f.keys()])
    data = f['data'][:]
    label = f['label'][:]
    ptsnum = f['ptsnum'][:]
    return (data, label, ptsnum)


def load_h5_bbox(h5_filename):
    f = h5py.File(h5_filename, 'r')
    bbox = f['bbox'][:]
    label = f['label'][:]
    return (bbox, label)
