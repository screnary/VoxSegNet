""" 201802 """
""" generate part bounding box """
import os
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import h5py
import warnings
import time
import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
import pdb

ThisDir = os.path.dirname(os.path.abspath(__file__))


class Visualize(object):
    """ a class for visualizing the 3d parts and their bouding boxes """

    def __init__(self, boxes, labels, parts_bag, Axl_m=-0.5, Axl_M=0.5, pro='3d'):
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection=pro)
        # ax.axis('equal')
        self.ax.set_aspect('equal')
        self.axl_m = Axl_m
        self.axl_M = Axl_M
        self.max_range = self.axl_M - self.axl_m
        self.ax.set_xlim(self.axl_m, self.axl_M)
        self.ax.set_ylim(self.axl_m, self.axl_M)
        self.ax.set_zlim(self.axl_m, self.axl_M)
        self.ax.set_xlabel('--X->')
        self.ax.set_ylabel('--Y->')
        self.ax.set_zlabel('--Z->')
        self.ax.view_init(elev=60., azim=-75.)
        self.Boxes = boxes
        self.Parts = parts_bag
        self.SubL = labels  # box labels(semantic)
        self.COLOR_PALETTE = np.loadtxt(ThisDir+'/palette_float.txt',dtype='float32')

    def __call__(self):
        print('visualize call fun')

    def drawCubes(self):
        """ Input Boxes: list of 8*3 np.array, 8 vertices """
        # list of sides' polygons of figure; cube surface
        print('Boxes num:', len(self.Boxes))
        for i in range(len(self.Boxes)):
            Z = self.Boxes[i]
            verts = [[Z[0],Z[1],Z[2],Z[3]],
                     [Z[4],Z[5],Z[6],Z[7]],
                     [Z[0],Z[1],Z[5],Z[4]],
                     [Z[2],Z[3],Z[7],Z[6]],
                     [Z[1],Z[2],Z[6],Z[5]],
                     [Z[4],Z[7],Z[3],Z[0]]]
            # plot sides
            collection = Poly3DCollection(verts,
                linewidths=.25, edgecolors='r', alpha=.25)
            # face_color = [0, 1, 1]
            face_color = self.COLOR_PALETTE[self.SubL[i].astype(int)]
            collection.set_facecolor(face_color)
            self.ax.add_collection3d(collection)

    def drawParts(self):
        c_idx = 0
        for p_idx in range(len(self.Parts)):
            part = self.Parts[p_idx]
            for i in range(part.shape[0]):
                sub_part = part[i]
                X = sub_part[:,0]
                Y = sub_part[:,1]
                Z = sub_part[:,2]
                p_color = self.COLOR_PALETTE[self.SubL[c_idx].astype(int)]
                c_idx = c_idx+1
                self.ax.scatter(X,Y,Z,c=p_color,alpha=0.25)
        print('Parts-withsub num:', c_idx)

    def disp(self, fname):
        # Create cubic bounding box to simulate equal aspect ratio
        Xb = 0.5*self.max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(self.axl_M+self.axl_m)
        Yb = 0.5*self.max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(self.axl_M+self.axl_m)
        Zb = 0.5*self.max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(self.axl_M+self.axl_m)
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            self.ax.plot([xb], [yb], [zb], 'w')
        # save and view
        self.fig.savefig(fname)
        # plt.show()


def min_set_dist(pts):
    ref = pts[0, ...]
    dists = np.sqrt(np.sum(np.square(pts - ref), axis=1))
    # dists_sort = np.sort(dists, axis=None)
    sort_index = np.argsort(dists, axis=None)
    dists_sort = dists[sort_index]
    ed = dists_sort.shape[0]
    if dists_sort.shape[0] > 11:
        ed = 11
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            metric = np.mean(dists_sort[1:ed])
        except RuntimeWarning as e:
            print('Catch RuntimeWarning:', e)
            pdb.set_trace()
    return metric, sort_index


def sets_dist(ptsSet1, ptsSet2):  # compute shortest distance between 2 sets
    if ptsSet1.shape[0] > ptsSet2.shape[0]:
        tmp = ptsSet1
        ptsSet1 = ptsSet2
        ptsSet2 = tmp

    mdist = np.amin(np.sqrt(np.sum(np.square(ptsSet2 - ptsSet1[0,:]), axis=1)))
    for pt in ptsSet1:
        dist = np.amin(np.sqrt(np.sum(np.square(ptsSet2 - pt), axis=1)))
        if dist < mdist:
            mdist = dist
    return mdist


def get_data_num(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    return data.shape[0]


def load_h5_volumes_data(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    seg = f['seg'][:]
    # bon = f['bon'][:]
    return (data, seg)


def load_h5_pts_data(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def scatter3D(pts):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = pts[:,0]
    Y = pts[:,1]
    Z = pts[:,2]
    ax.scatter(X,Y,Z,alpha=0.8)
    # ax.set_aspect('equal')
    ax.axis('equal')
    ax.set_xlim(0.,1.)
    ax.set_ylim(0.,1.)
    ax.set_zlim(0.,1.)
    ax.set_xlabel('--X->')
    ax.set_ylabel('--Y->')
    ax.set_zlabel('--Z->')
    ax.view_init(elev=60., azim=-75.)


def computeBox(pts, plb, alpha=1.5):
    """ compute Boxes given model point cloud and point label
        alpha=3.0
    """
    metric, _ = min_set_dist(pts)  # min Euclidian distance of the point set
    alpha = alpha  # scale factor of metric
    label_tab = np.unique(plb)

    parts_bag = []
    label_bag = []
    for l in label_tab:
        idx = plb == l
        part_p_ = pts[idx, ...]
        # pdb.set_trace()
        # print('proccessing part label:', str(l))
        if part_p_.shape[0] < 3:
            continue
        _, sort_idx = min_set_dist(part_p_)
        part_p = part_p_[sort_idx, ...].tolist()

        que = []  # subpart que
        que.append(part_p.pop(0))
        queSet = []
        queSet.append(que)

        i = 0
        """ generate sub part sets """
        while len(part_p) > 0:
            # for the point part_p[i]

            for q_idx in range(len(queSet)):
                # for_loop not recommend for list operation (del / append)
                # for the subset queSet[q_idx]
                if i == len(part_p):
                    """ i == len(part_p), all points checked """
                    # print('subset num:', str(len(queSet)), '  remain pts num:', str(len(part_p)))
                    i = 0  # restart checking part_p

                    # after checking all subparts in queSet,
                    # pt is not belong to cur subparts
                    """ re-sort part points w.r.t dist """
                    if len(part_p) > 1:
                        part_p_ = np.asarray(part_p)
                        _, sort_idx = min_set_dist(part_p_)
                        part_p = part_p_[sort_idx, ...].tolist()
                    """ creat new subset """
                    que_tmp = []
                    que_tmp.append(part_p.pop(0))
                    queSet.append(que_tmp)
                    break
                pt = part_p[i]
                # the min distance from pt to cur pts_set
                d_h = directed_hausdorff(np.expand_dims(pt, 0), np.asarray(queSet[q_idx]))[0]
                if d_h < metric*alpha:
                    # the point within the cur set region
                    queSet[q_idx].append(pt)
                    del part_p[i]
                    break  # no need to check the other subsets
                elif q_idx < len(queSet) - 1:
                    continue
                else:
                    # skip when the point not belong to all current subset
                    i = i + 1

        """ merge subsets if possible """
        subPart = []
        subPart.append(queSet.pop(0))
        n_subp0 = len(queSet) + 1
        n_subp1 = len(queSet)

        while len(queSet) > 0:
            if n_subp0 == n_subp1:
                subPart.append(queSet.pop(0))
                n_subp1 = len(queSet)
            n_subp0 = len(queSet)
            for j in range(len(subPart)):
                d_idx = []
                for k in range(len(queSet)):
                    set_dist = sets_dist(np.asarray(subPart[j]), np.asarray(queSet[k]))
                    if set_dist < metric*alpha:
                        subPart[j].extend(queSet[k])
                        d_idx.append(k)  # the subset index should be merged into subPart[j]
                # print('queSet-len:'+str(len(queSet)))
                # print('subPart-len:'+str(len(subPart)))
                # print('del-idx:', d_idx)
                # pdb.set_trace()
                for kk in range(len(d_idx)-1, -1, -1):
                    del queSet[d_idx[kk]]  # d_idx increase, so del from tail
                n_subp1 = len(queSet)

        """ remove noise subsets """
        subnum = np.asarray([len(sub) for sub in subPart])
        tol = np.min([np.mean(subnum), np.median(subnum)])/20  # this should be changed for Airplane (small wheel)
        valid_idx = subnum >= tol
        # print('subpart num:', subnum)
        # print('valid thread:', tol)
        # print('valid flag:', valid_idx)
        ptsSets = np.asarray([np.asarray(sub) for sub in subPart])
        # pdb.set_trace()
        parts_bag.append(ptsSets[valid_idx, ...])
        label_bag.append(l)

    """ generate bounding boxes """
    Boxes = []
    BLabels = []  # box label
    SubParts = []
    SPlabels = []

    for n in range(len(parts_bag)):
        part = parts_bag[n]
        label = label_bag[n]
        for i in range(part.shape[0]):
            Box = []
            sub_part = part[i]  # np array shape=(num_pts, 3)
            minCoord = np.amin(sub_part, axis=0)
            maxCoord = np.amax(sub_part, axis=0)

            x_m = minCoord[0]
            y_m = minCoord[1]
            z_m = minCoord[2]
            x_M = maxCoord[0]
            y_M = maxCoord[1]
            z_M = maxCoord[2]

            # for extreamly small widgets
            h = x_M - x_m + 1
            w = y_M - y_m + 1
            d = z_M - z_m + 1
            if h*w*d < 2:
                continue
            elif h==1 or w==1 or d==1:
                if h==1:
                    x_m = max(x_m - 1, 0)
                    x_M = min(x_M + 1, 47)  # vox size 48
                if w==1:
                    y_m = max(y_m - 1, 0)
                    y_M = min(y_M + 1, 47)
                if d==1:
                    z_m = max(z_m - 1, 0)
                    z_M = min(z_M + 1, 47)

            # 8 vertices (in order)
            #   H---G
            # D---C |
            # | E-|-F
            # A---B
            Box.append([x_m, y_m, z_m])  # A
            Box.append([x_M, y_m, z_m])  # B
            Box.append([x_M, y_m, z_M])  # C
            Box.append([x_m, y_m, z_M])  # D
            Box.append([x_m, y_M, z_m])  # E
            Box.append([x_M, y_M, z_m])  # F
            Box.append([x_M, y_M, z_M])  # G
            Box.append([x_m, y_M, z_M])  # H

            Boxes.append(np.asarray(Box))
            BLabels.append(label)
            SubParts.append(sub_part)
            SPlabels.append(label)

    return Boxes, BLabels, SubParts, SPlabels


def gen_rois_info(Boxes, Blabels):
    """ vox version: compute from boxes
        mask_shape: mini_mask_shape
        mimic minimize_mask() in MASK_RCNN project
            use interpolation, mask smooth
        get proposal voxel in roi, also (mini_voxels)
    """
    boxes = []
    part_ids = []
    for n in range(len(Blabels)):
        # get part_instance full size mask
        label = Blabels[n]
        box = Boxes[n]
        minCoord = box[0, :]
        maxCoord = box[-2, :]
        x_m = minCoord[0].astype(int)  # x,y,z is r,c,d
        y_m = minCoord[1].astype(int)
        z_m = minCoord[2].astype(int)
        x_M = maxCoord[0].astype(int)
        y_M = maxCoord[1].astype(int)
        z_M = maxCoord[2].astype(int)

        bbox = np.array([x_m,y_m,z_m,x_M,y_M,z_M])

        boxes.append(bbox)  # pred_bbox
        part_ids.append(label)  # start from 1

    return boxes, part_ids


def augment_rois(boxes, part_ids, vox_size):
    """ expand, translate, rotate
        boxes: (num_ins, (x1,y1,z1,x2,y2,z2))
    """
    boxes = np.array(boxes).astype(np.float32)
    part_ids = np.array(part_ids).astype(np.float32)
    centers = (boxes[:, 0:3] + boxes[:, 3:6]) / 2
    lxs = boxes[:,3] - boxes[:,0]
    lys = boxes[:,4] - boxes[:,1]
    lzs = boxes[:,5] - boxes[:,2]
    # Expand:
    scale = 2
    boxes_expand = np.squeeze(np.array([
        [centers[:,0]-lxs/2-lxs/scale], [centers[:,1]-lys/2-lys/scale], [centers[:,2]-lzs/2-lzs/scale],
        [centers[:,0]+lxs/2+lxs/scale], [centers[:,1]+lys/2+lys/scale], [centers[:,2]+lzs/2+lzs/scale]])).T
    # pdb.set_trace()
    # print('boxes_expand shape: ', boxes_expand.shape)
    if len(boxes_expand.shape)<2:
        boxes_expand = np.expand_dims(boxes_expand, axis=0)
    boxes_expand[boxes_expand < 0.0] = 0.0
    boxes_expand[boxes_expand > vox_size-1] = vox_size-1
    boxes_expand = np.round(boxes_expand)
    pids_expand = part_ids
    # Rotate:

    boxes_aug = np.concatenate([boxes,boxes_expand],0)
    partids_aug = np.concatenate([part_ids,pids_expand],0)
    return boxes_aug, partids_aug


def augment_gt_rois(boxes, partids, vox_size):
    """ augment by random sample boxes, not normalized
    """
    scale_range = np.array([0.5, 1.25])  # box size
    locat_range = np.array([0.0, 1.0])  # box location
    batch_size = boxes.shape[0]
    rois_num = boxes.shape[1]
    np.random.seed(int(time.time()))
    boxes_new = np.zeros(boxes.shape)
    partids_new = np.zeros(partids.shape)
    for b in range(batch_size):
        rois = boxes[b, ...]
        pids = partids[b, ...]
        valid_idx = pids != 0
        rois = rois[valid_idx, ...]
        pids = pids[valid_idx, ...]
        corner = rois[:, 0:3]
        side_len = rois[:,3:] - rois[:,0:3]  # [lxs,lys,lzs]
        scale_factor = (scale_range[1]-scale_range[0]) *\
            np.random.random_sample((rois.shape[0],3)) + scale_range[0]
        locat_factor = (locat_range[1]-locat_range[0]) *\
            np.random.random_sample((rois.shape[0],3)) + locat_range[0]
        side_new = side_len * scale_factor
        side_new[side_new<=1] = 2.0  # ensure reasonable box
        centers = corner + side_len*locat_factor
        lxs = side_new[:,0]
        lys = side_new[:,1]
        lzs = side_new[:,2]
        rois_new = np.squeeze(np.array([
            [centers[:,0]-lxs/2], [centers[:,1]-lys/2], [centers[:,2]-lzs/2],
            [centers[:,0]+lxs/2], [centers[:,1]+lys/2], [centers[:,2]+lzs/2]])).T
        rois_new[rois_new < 0.0] = 0.0
        rois_new[rois_new > vox_size-1] = vox_size-1
        rois_new = rois_new.round()
        if len(rois_new.shape) == 1:
            rois_new = np.expand_dims(rois_new,0)
        try:
            rois_aug = np.concatenate([rois, rois_new], 0)
            pids_aug = np.concatenate([pids, pids], 0)
        except ValueError as e:
            pdb.set_trace()
        b_count = rois_aug.shape[0]
        if b_count <= rois_num:
            boxes_new[b,0:b_count,...] = rois_aug
            partids_new[b,0:b_count,...] = pids_aug
        else:
            boxes_new[b,...] = rois_aug[0:rois_num,...]
            partids_new[b,...] = pids_aug[0:rois_num,...]
    return boxes_new, partids_new


def augment_rand_rois(boxes, partids, vox_size, max_ins_num):
    """ augment by random sample boxes, not normalized
    """
    scale_range = np.array([0.5, 1.25])  # box size
    locat_range = np.array([0.0, 1.0])  # box location
    batch_size = boxes.shape[0]
    rois_num = max_ins_num
    np.random.seed(int(time.time()))
    boxes_new = np.zeros([batch_size, rois_num, 6])
    partids_new = np.zeros([batch_size, rois_num])
    for b in range(batch_size):
        rois = boxes[b, ...]
        pids = partids[b, ...]
        valid_idx = pids != 0
        rois = rois[valid_idx, ...]
        pids = pids[valid_idx, ...]
        corner = rois[:, 0:3]
        side_len = rois[:,3:] - rois[:,0:3]  # [lxs,lys,lzs]
        scale_factor = (scale_range[1]-scale_range[0]) *\
            np.random.random_sample((rois.shape[0],3)) + scale_range[0]
        locat_factor = (locat_range[1]-locat_range[0]) *\
            np.random.random_sample((rois.shape[0],3)) + locat_range[0]
        side_new = side_len * scale_factor
        side_new[side_new<=1] = 2.0  # ensure reasonable box
        centers = corner + side_len*locat_factor
        lxs = side_new[:,0]
        lys = side_new[:,1]
        lzs = side_new[:,2]
        rois_new = np.squeeze(np.array([
            [centers[:,0]-lxs/2], [centers[:,1]-lys/2], [centers[:,2]-lzs/2],
            [centers[:,0]+lxs/2], [centers[:,1]+lys/2], [centers[:,2]+lzs/2]])).T
        rois_new[rois_new < 0.0] = 0.0
        rois_new[rois_new > vox_size-1] = vox_size-1
        rois_new = rois_new.round()
        if len(rois_new.shape) == 1:
            rois_new = np.expand_dims(rois_new,0)
        try:
            rois_aug = np.concatenate([rois, rois_new], 0)
            pids_aug = np.concatenate([pids, pids], 0)
        except ValueError as e:
            pdb.set_trace()
        b_count = rois_aug.shape[0]
        if b_count <= rois_num:
            boxes_new[b,0:b_count,...] = rois_aug
            partids_new[b,0:b_count,...] = pids_aug
        else:
            boxes_new[b,...] = rois_aug[0:rois_num,...]
            partids_new[b,...] = pids_aug[0:rois_num,...]
    return boxes_new, partids_new


def augment_pred_rois(boxes, partids, vox_size, max_ins_num):
    """ augment by random sample boxes, not normalized
    """
    # scale_range = np.array([0.5, 1.25])  # box size
    # locat_range = np.array([0.0, 1.0])  # box location
    batch_size = boxes.shape[0]
    rois_num = max_ins_num
    # np.random.seed(int(time.time()))
    boxes_new = np.zeros([batch_size, rois_num, 6])
    partids_new = np.zeros([batch_size, rois_num])
    # print('boxes_new shape:',boxes_new.shape)
    for b in range(batch_size):
        rois = boxes[b, ...]
        pids = partids[b, ...]
        valid_idx = pids != 0
        rois = rois[valid_idx, ...]
        pids = pids[valid_idx, ...]
        corner = rois[:, 0:3]
        side_len = rois[:,3:] - rois[:,0:3]  # [lxs,lys,lzs]
        # sort the rois and pids according to areas
        # from large to small
        areas = side_len[:,0]*side_len[:,1]*side_len[:,2]
        sortidx = np.argsort(-areas)
        rois = rois[sortidx,...]
        pids = pids[sortidx,...]
        side_len = side_len[sortidx,...]
        axis_max_idx = np.argmax(side_len,axis=-1)  # the longest axis of rois
        axis_min_idx = np.argmin(side_len,axis=-1)
        # construct scale and location factors
        rois_ = np.repeat(rois,2,axis=0)
        pids_ = np.repeat(pids,2,axis=0)
        side_len_ = np.repeat(side_len,2,axis=0)
        corner_ = np.repeat(corner,2,axis=0)
        num_aug = rois_.shape[0]
        scale_factor = np.ones((num_aug,3)) * 0.5
        scale_factor[range(0,num_aug,2),axis_min_idx] = 1.0
        scale_factor[range(1,num_aug,2),axis_min_idx] = 1.0
        locat_factor = np.ones((num_aug,3)) * 0.5
        locat_factor[range(0,num_aug,2),axis_max_idx] = 0.25
        locat_factor[range(1,num_aug,2),axis_max_idx] = 0.75
        # scale_factor = (scale_range[1]-scale_range[0]) *\
        #     np.random.random_sample((rois.shape[0],3)) + scale_range[0]
        # locat_factor = (locat_range[1]-locat_range[0]) *\
        #     np.random.random_sample((rois.shape[0],3)) + locat_range[0]
        side_new = side_len_ * scale_factor
        side_new[side_new<=1] = 2.0  # ensure reasonable box
        centers = corner_ + side_len_*locat_factor
        lxs = side_new[:,0]
        lys = side_new[:,1]
        lzs = side_new[:,2]
        rois_new = np.squeeze(np.array([
            [centers[:,0]-lxs/2], [centers[:,1]-lys/2], [centers[:,2]-lzs/2],
            [centers[:,0]+lxs/2], [centers[:,1]+lys/2], [centers[:,2]+lzs/2]])).T
        rois_new[rois_new < 0.0] = 0.0
        rois_new[rois_new > vox_size-1] = vox_size-1
        rois_new = rois_new.round()
        if len(rois_new.shape) == 1:
            rois_new = np.expand_dims(rois_new,0)
        try:
            rois_aug = np.concatenate([rois, rois_new], 0)
            pids_aug = np.concatenate([pids, pids_], 0)
        except ValueError as e:
            pdb.set_trace()
        b_count = rois_aug.shape[0]
        if b_count <= rois_num:
            boxes_new[b,0:b_count,...] = rois_aug
            partids_new[b,0:b_count,...] = pids_aug
        else:
            boxes_new[b,...] = rois_aug[0:rois_num,...]
            partids_new[b,...] = pids_aug[0:rois_num,...]
        # print('valid boxes num:', np.sum(np.sum(np.squeeze(boxes_new),axis=-1)>0))
        # print('valid partids num:', np.sum(np.squeeze(partids_new)>0))
        # print('valid rois_aug num:', np.sum(np.sum(np.squeeze(rois_aug),axis=-1)>0))
        # print('valid pids_aug num:', np.sum(np.squeeze(pids_aug)>0))
        # print('rois_aug:',rois_aug)
    return boxes_new, partids_new


if __name__ == '__main__':
    """ Settings """
    Veaw_Flag = True

    DATA_DIR = os.path.join('/media/screnary', 'E/LabWork/Data', 'PointsData/hdf5_data')
    # CAT_NAME = 'Motorbike'  # ['Motorbike', 'Earphone', 'Rocket', 'Airplane', 'Chair']
    CAT_LIST = ['Motorbike', 'Earphone', 'Rocket', 'Airplane', 'Chair']
    dataType = 'vol'  # 'pts'
    train_or_test = 'test'

    dump_dir = './cubeImgs/'
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
    # pdb.set_trace()
    """ input data """
    for CAT_NAME in CAT_LIST:
        print('Category: ', CAT_NAME)
        out_dir = os.path.join('./cubeImgs/', CAT_NAME+'/')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        m_file = os.path.join(DATA_DIR, CAT_NAME+'_'+dataType+'_'+train_or_test+'.h5')
        vox_data, _ = load_h5_volumes_data(m_file)
        model_num = vox_data.shape[0]
        # m_idx = 2  # model idx
        for m_idx in range(model_num):
            print('-----proccessing model:', m_idx, ' of ', model_num-1, '-----')
            if dataType == 'pts':
                current_pts, current_plb = load_h5_pts_data(m_file)
                pts, indices = np.unique(current_pts[m_idx], return_index=True, axis=0)
                plb = current_plb[m_idx][indices]
                axl_m = -0.5
                axl_M = 0.5
            elif dataType == 'vol':
                vox_data, vox_seg = load_h5_volumes_data(m_file)
                cur_vox = np.squeeze(vox_data[m_idx, ...])
                cur_seg = np.squeeze(vox_seg[m_idx, ...])
                pts = np.transpose(np.array(np.where(cur_vox)))
                plb = cur_seg[pts[:,0], pts[:,1], pts[:,2]]
                pts = pts.astype(float)
                axl_m = -1.
                axl_M = 48.
            # pdb.set_trace()
            """ process seg pts into subpart groups (distance metric, group box)"""
            Boxes, BLabels, parts_bag, _ = computeBox(pts=pts, plb=plb, alpha=1.5)

            # pdb.set_trace()

            """ visualize bounding effect """
            if Veaw_Flag:
                plt.close('all')
                Visbox = Visualize(boxes=Boxes, labels=BLabels, parts_bag=parts_bag, Axl_m=axl_m, Axl_M=axl_M)
                # Visbox()
                # pdb.set_trace()
                Visbox.drawCubes()
                Visbox.disp(out_dir+CAT_NAME+'_'+dataType+'_'+train_or_test+str(m_idx)+'_cubes.png')
                del Visbox

                Vispart = Visualize(boxes=Boxes, labels=BLabels, parts_bag=parts_bag, Axl_m=axl_m, Axl_M=axl_M)
                Vispart.drawParts()
                Vispart.disp(out_dir+CAT_NAME+'_'+dataType+'_'+train_or_test+str(m_idx)+'_parts.png')
                del Vispart

            # Vis = Visualize(boxes=Boxes, labels=BLabels, parts_bag=parts_bag, Axl_m=axl_m, Axl_M=axl_M)
            # Vis.drawParts()
            # Vis.drawCubes()
            # Vis.disp(out_dir+CAT_NAME+'_'+dataType+'_'+train_or_test+str(m_idx)+'_part_cube.png')
            # del Vis
            # pdb.set_trace()
