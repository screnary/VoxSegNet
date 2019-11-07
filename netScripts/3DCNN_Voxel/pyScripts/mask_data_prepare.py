""" preparing mask data for mask prediction net --- 20180313 """
""" mask gt: [instance, height, width]
    bbox gt: [instance, (x1,y1,z1, x2,y2,z2)]
        {this x,y,z is for 3D tensor, min and max corner}
    part_ids: 1D array of part IDs of the part mask and bbox, start from 1
    padding 0 for non-exist part
    Max_Instance_Num set to be 15
"""
import os
import sys
import numpy as np
from scipy.interpolate import griddata
import h5py
import warnings
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
import pdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

Common_SET_DIR = os.path.join(BASE_DIR, '../CommonFile')
sys.path.append(Common_SET_DIR)
import globals as g_

import gen_part_box as Box

COLOR_PALETTE = np.loadtxt(os.path.join(BASE_DIR,
    'utils/palette_float.txt'),dtype='float32')


def tic():
    globals()['tt'] = time.clock()


def toc():
    print('\nElapsed time: %.8f seconds' % (time.clock()-globals()['tt']))


def pts_to_volume(points, vsize):
    """ input is Nx3 points. (normalized)
        output is vsize*vsize*vsize
        assumes points are in range [0, 1]
        (actually, after nomalization, all points are in a unit sphere)
    """
    # make sure each point on grid point [0,1,2,3,4] for vsize 5
    shift = -0.5
    vol = np.zeros((vsize,vsize,vsize))
    voxel = 1.0/float(vsize)
    locations = points/voxel
    locations = np.round(locations+shift).astype(int)  # np.floor().astype(int)
    # dealing with box boundary
    # locations[locations > (vsize - 1)] = vsize - 1
    # locations[locations < 0] = 0
    vol[locations[:,0],locations[:,1],locations[:,2]] = 1.0
    return vol


def gen_masks_info_v1(parts_bag, label_bag, mask_shape=5):
    """ pts version: compute from parts_bag (points bag)
        parts_bag: (n_l) list; label_bag: (n_l) list
        boxes: (n,6) np.array; labels: (n,1); masks: (n, MASK_SHAPE)
        MASK_SHAPE = (5,5,5)
            not use interpolation, mask could be not good
    """
    # boxes = []
    # masks = []
    # part_ids = []
    part_pts = []  # part points group
    for n in range(len(parts_bag)):
        part = parts_bag[n]  # for this label, sub_parts in one bag: part
        # label = label_bag[n]
        for i in range(part.shape[0]):
            # box = []
            sub_part = part[i]  # np.array shape=(num_pts, 3)
            # minCoord = np.amin(sub_part, axis=0)
            # maxCoord = np.amax(sub_part, axis=0)
            # box.extend(minCoord)
            # box.extend(maxCoord)

            # # sub_part --> mini_mask
            # """ normalize """
            # pts = (sub_part - minCoord)/(maxCoord - minCoord)
            # mask = pts_to_volume(pts, mask_shape)

            # # save the result info
            # boxes.append(box)
            # masks.append(mask)
            # part_ids.append(label)
            part_pts.append(sub_part)
    return part_pts
    # return masks, boxes, part_ids, part_pts


def cropping_mask(bbox, mask):
    """ crop mask """
    x_m = bbox[0].astype(int)
    y_m = bbox[1].astype(int)
    z_m = bbox[2].astype(int)
    x_M = bbox[3].astype(int)
    y_M = bbox[4].astype(int)
    z_M = bbox[5].astype(int)
    m = mask[x_m:(x_M+1), y_m:(y_M+1), z_m:(z_M+1)]  # crop mask
    if m.size == 0:
        raise Exception("Invalid bounding box with area of zero")

    return m


def minimize_mask(bbox, mask, mini_shape):
    """ crop and resize to MASK_SHAPE, minimize_mask()
    Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()
    """
    m = cropping_mask(bbox, mask)
    h,w,d = m.shape
    maskAux = m.reshape(h*w*d)
    # ijk is an (h*w*d, 3) array with the indexes of the reshaped array
    ijk = np.mgrid[0:h, 0:w, 0:d].reshape(3, h*w*d).T
    # interpolate position num
    n_in = complex(0, mini_shape)
    i,j,k = np.mgrid[0:(h-1):n_in, 0:(w-1):n_in, 0:(d-1):n_in]
    # method could be "nearest", "linear", or "cubic"
    mini_mask = griddata(ijk, maskAux, (i,j,k), method="linear") >= 0.5

    return mini_mask


def expand_mask(bbox, mini_mask, vox_shape):
    """Resizes mini masks back to voxel size. Reverses the change
    of minimize_mask().
    return mask : [vox_shape, vox_shape, vox_shape], {0,1} value
    """
    h,w,d = mini_mask.shape
    mini_maskAux = mini_mask.reshape(h*w*d)
    # ijk is an (h*w*d, 3) array with the indexes of the reshaped array
    ijk = np.mgrid[0:h, 0:w, 0:d].reshape(3, h*w*d).T
    # interpolate
    x_m = bbox[0].astype(int)
    y_m = bbox[1].astype(int)
    z_m = bbox[2].astype(int)
    x_M = bbox[3].astype(int)
    y_M = bbox[4].astype(int)
    z_M = bbox[5].astype(int)
    b_h = x_M - x_m + 1
    b_w = y_M - y_m + 1
    b_d = z_M - z_m + 1
    n_in = [complex(0, b_h), complex(0, b_w), complex(0, b_d)]
    i,j,k = np.mgrid[0:(h-1):n_in[0], 0:(w-1):n_in[1], 0:(d-1):n_in[2]]
    # method could be "nearest", "linear", or "cubic"
    mask_cropped = griddata(ijk, mini_maskAux, (i,j,k), method="linear") >= 0.5

    mask = np.zeros([vox_shape, vox_shape, vox_shape])
    mask[x_m:(x_M+1), y_m:(y_M+1), z_m:(z_M+1)] = mask_cropped

    return mask, mask_cropped


def gen_masks_info_v2(vox_seg, Boxes, Blabels, mask_shape=16):
    """ vox version: compute from boxes
        mask_shape: mini_mask_shape
        mimic minimize_mask() in MASK_RCNN project
            use interpolation, mask smooth
    """
    boxes = []
    masks = []
    masks_crop = []
    mini_masks = []
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
        embed_box = np.zeros(vox_seg.shape)
        embed_box[x_m:(x_M+1), y_m:(y_M+1), z_m:(z_M+1)] = 1
        mask = np.multiply((vox_seg == label), embed_box)  # elementwise, same as *
        mask_pts = np.transpose(np.array(np.where(mask)))

        bbox = np.array([x_m,y_m,z_m,x_M,y_M,z_M])

        # print(n)
        # if n == 8:
        #     pdb.set_trace()
        # crop and resize to MASK_SHAPE, minimize_mask()
        mini_mask = minimize_mask(bbox, mask, mask_shape)

        boxes.append(bbox)
        masks.append(mask_pts)
        masks_crop.append(cropping_mask(bbox, mask))
        mini_masks.append(mini_mask)
        part_ids.append(label)  # start from 1

    return masks_crop, mini_masks, boxes, part_ids


def gen_masks_info_v3(vox_seg, Boxes, Blabels, mask_shape=16):
    """ vox version: compute from boxes
        mask_shape: mini_mask_shape
        mimic minimize_mask() in MASK_RCNN project
            use interpolation, mask smooth
        get proposal voxel in roi, also (mini_voxels)
    """
    boxes = []
    mini_masks = []
    mini_voxels = []
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
        embed_box = np.zeros(vox_seg.shape)
        embed_box[x_m:(x_M+1), y_m:(y_M+1), z_m:(z_M+1)] = 1
        mask = np.multiply((vox_seg == label), embed_box)  # elementwise, same as *
        voxel = np.multiply((vox_seg > 0), embed_box)
        # voxel_pts = np.transpose(np.array(np.where(voxel)))

        bbox = np.array([x_m,y_m,z_m,x_M,y_M,z_M])
        # pdb.set_trace()
        # print(n)
        # if n == 8:
        #     pdb.set_trace()
        # crop and resize to MASK_SHAPE, minimize_mask()
        mini_mask = minimize_mask(bbox, mask, mask_shape)
        mini_voxel = minimize_mask(bbox, voxel, mask_shape)

        boxes.append(bbox)  # pred_bbox
        mini_masks.append(mini_mask)  # target_masks
        mini_voxels.append(mini_voxel)  # proposals
        part_ids.append(label)  # start from 1

    return mini_masks, mini_voxels, boxes, part_ids


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


def draw_pts(part_pts):
    """ for visualization, quick """
    # print('drawing mask...')
    plt.close('all')

    fig = plt.figure()
    # plot the mini_mask
    # ax = fig.gca(projection='3d'

    # plot the full size mask: in pts format
    ax_1 = fig.add_subplot(111, projection='3d')
    ax_1.set_title('tfgraph computed box')
    ax_1.set_aspect('equal')
    ax_1.set_xlabel('--X->')
    ax_1.set_ylabel('--Y->')
    ax_1.set_zlabel('--Z->')
    # ax_1.view_init(elev=79., azim=-65.)
    ax_1.view_init(elev=17., azim=165.)
    # pdb.set_trace()

    X_pts = part_pts[:,0]
    Y_pts = part_pts[:,1]
    Z_pts = part_pts[:,2]

    # axl_m = np.amin(BOXES, axis=0)-1
    # axl_M = np.amax(BOXES, axis=0)+1
    # max_range = axl_M - axl_m
    # axl_m = np.min(BOXES)-1
    # axl_M = np.max(BOXES)+1
    axl_m = np.min(part_pts)
    axl_M = np.max(part_pts)
    ax_1.set_xlim(axl_m,axl_M)
    ax_1.set_ylim(axl_m,axl_M)
    ax_1.set_zlim(axl_m,axl_M)
    ax_1.scatter(X_pts,Y_pts,Z_pts,alpha=0.8)


def draw_voxel(voxel_data, ax):
    '''
    =============
    3D voxel plot
    =============
    '''

    def explode(data):
        # explode voxel data
        size = np.array(data.shape)*2
        # insert gaps between valid grid
        data_e = np.zeros(size - 1, dtype=data.dtype)
        data_e[::2, ::2, ::2] = data
        return data_e

    # build up the numpy logo
    facecolors = np.where(voxel_data, '#FFD65DC0', '#7A88CC00')  # '#7A88CCC0'
    edgecolors = np.where(voxel_data, '#BFAB6E', '#7D84A600')  # '#7D84A6'
    filled = np.ones(voxel_data.shape)

    # upscale the above voxel image, leaving gaps
    filled_2 = explode(filled)
    fcolors_2 = explode(facecolors)
    ecolors_2 = explode(edgecolors)
    # pdb.set_trace()

    # Shrink the gaps, gap size = 0.1
    x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.05
    y[:, 0::2, :] += 0.05
    z[:, :, 0::2] += 0.05
    x[1::2, :, :] += 0.95
    y[:, 1::2, :] += 0.95
    z[:, :, 1::2] += 0.95

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=ecolors_2)


def draw_mask(mini_mask, full_mask):
    # print('drawing mask...')
    plt.close('all')

    fig = plt.figure()
    # plot the mini_mask
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title('mini mask')
    axl_m = 0
    axl_M = np.max(mini_mask.shape)
    ax.set_aspect('equal')
    ax.set_xlim(axl_m,axl_M)
    ax.set_ylim(axl_m,axl_M)
    ax.set_zlim(axl_m,axl_M)
    ax.set_xlabel('--X->')
    ax.set_ylabel('--Y->')
    ax.set_zlabel('--Z->')
    ax.view_init(elev=79., azim=-65.)
    draw_voxel(mini_mask, ax)

    # plot the full size mask
    ax_1 = fig.add_subplot(122, projection='3d')
    ax_1.set_title('crop mask')
    ax_1.set_aspect('equal')
    ax_1.set_xlabel('--X->')
    ax_1.set_ylabel('--Y->')
    ax_1.set_zlabel('--Z->')
    ax_1.view_init(elev=79., azim=-65.)
    if len(full_mask.shape) == 2:
        part_pts = full_mask
        X_pts = part_pts[:,0]
        Y_pts = part_pts[:,1]
        Z_pts = part_pts[:,2]

        axl_m = -1
        axl_M = 48
        ax_1.set_xlim(axl_m,axl_M)
        ax_1.set_ylim(axl_m,axl_M)
        ax_1.set_zlim(axl_m,axl_M)
        ax_1.scatter(X_pts,Y_pts,Z_pts,alpha=0.8)
    elif len(full_mask.shape) == 3:
        # print('Warning: full_mask should be pts format, vox is too costly!')
        axl_m = 0
        axl_M = np.max(full_mask.shape)
        ax_1.set_xlim(axl_m,axl_M)
        ax_1.set_ylim(axl_m,axl_M)
        ax_1.set_zlim(axl_m,axl_M)
        draw_voxel(full_mask, ax_1)  # mask crop


def draw_mask_v2(mini_mask, full_mask, box_bag, part_ids):
    """ for visualization, quick """
    # print('drawing mask...')
    plt.close('all')

    fig = plt.figure()
    # plot the mini_mask
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title('mini mask')
    axl_m = 0
    axl_M = np.max(mini_mask.shape)
    ax.set_aspect('equal')
    ax.set_xlim(axl_m,axl_M)
    ax.set_ylim(axl_m,axl_M)
    ax.set_zlim(axl_m,axl_M)
    ax.set_xlabel('--X->')
    ax.set_ylabel('--Y->')
    ax.set_zlabel('--Z->')
    ax.view_init(elev=41., azim=-64.)
    draw_voxel(mini_mask, ax)

    # plot the full size mask: in pts format
    ax_1 = fig.add_subplot(122, projection='3d')
    ax_1.set_title('full mask')
    ax_1.set_aspect('equal')
    ax_1.set_xlabel('--X->')
    ax_1.set_ylabel('--Y->')
    ax_1.set_zlabel('--Z->')
    # ax_1.view_init(elev=79., azim=-65.)
    ax_1.view_init(elev=41., azim=-64.)
    # pdb.set_trace()
    if len(full_mask.shape) == 1:
        full_mask = np.concatenate(full_mask,0)
    if len(full_mask.shape) == 2:
        part_pts = full_mask
        X_pts = part_pts[:,0]
        Y_pts = part_pts[:,1]
        Z_pts = part_pts[:,2]

        BOXES = np.asarray(box_bag).reshape([-1,3])
        # axl_m = np.amin(BOXES, axis=0)-1
        # axl_M = np.amax(BOXES, axis=0)+1
        # max_range = axl_M - axl_m
        axl_m = np.min(BOXES)-1
        axl_M = np.max(BOXES)+1
        max_range = axl_M - axl_m
        ax_1.set_xlim(axl_m,axl_M)
        ax_1.set_ylim(axl_m,axl_M)
        ax_1.set_zlim(axl_m,axl_M)
        ax_1.scatter(X_pts,Y_pts,Z_pts,alpha=0.8)
    elif len(full_mask.shape) == 3:
        # print('Warning: full_mask should be pts format, vox is too costly!')
        axl_m = 0
        axl_M = np.max(full_mask.shape)
        max_range = axl_M - axl_m
        ax_1.set_xlim(axl_m,axl_M)
        ax_1.set_ylim(axl_m,axl_M)
        ax_1.set_zlim(axl_m,axl_M)
        draw_voxel(full_mask, ax_1)  # mask crop

    # draw the bounding boxes of the whole shape
    def drawCubes(Boxes, id_bag, ax_in):
        """ Input Boxes: list of 8*3 np.array, 8 vertices """
        # list of sides' polygons of figure; cube surface
        print('Boxes num:', len(Boxes))
        for i in range(len(Boxes)):
            Z = Boxes[i]
            verts = [[Z[0],Z[1],Z[2],Z[3]],
                     [Z[4],Z[5],Z[6],Z[7]],
                     [Z[0],Z[1],Z[5],Z[4]],
                     [Z[2],Z[3],Z[7],Z[6]],
                     [Z[1],Z[2],Z[6],Z[5]],
                     [Z[4],Z[7],Z[3],Z[0]]]
            # plot sides
            collection = Poly3DCollection(verts,
                linewidths=.25, edgecolors='r', alpha=.15)
            # face_color = [0, 1, 1]
            face_color = COLOR_PALETTE[id_bag[i].astype(int)]
            collection.set_facecolor(face_color)
            ax_in.add_collection3d(collection)

    drawCubes(box_bag, part_ids, ax_1)

    # Create cubic bounding box to simulate equal aspect ratio
    # pdb.set_trace()
    try:
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2].flatten() + 0.5*(axl_M+axl_m)
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2].flatten() + 0.5*(axl_M+axl_m)
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2].flatten() + 0.5*(axl_M+axl_m)
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax_1.plot([xb], [yb], [zb], 'w')
    except UnboundLocalError as e:
        print('Error! ', e)
        pdb.set_trace()


def draw_mask_box(full_mask, box_bag, part_ids):
    """ for visualization, quick """
    # print('drawing mask...')
    plt.close('all')

    fig = plt.figure()
    # plot the mini_mask
    # ax = fig.gca(projection='3d'

    # plot the full size mask: in pts format
    ax_1 = fig.add_subplot(111, projection='3d')
    ax_1.set_title('tfgraph computed box')
    ax_1.set_aspect('equal')
    ax_1.set_xlabel('--X->')
    ax_1.set_ylabel('--Y->')
    ax_1.set_zlabel('--Z->')
    # ax_1.view_init(elev=79., azim=-65.)
    ax_1.view_init(elev=41., azim=-64.)
    # pdb.set_trace()
    if len(full_mask.shape) == 1:
        full_mask = np.concatenate(full_mask,0)
    if len(full_mask.shape) == 2:
        part_pts = full_mask
        X_pts = part_pts[:,0]
        Y_pts = part_pts[:,1]
        Z_pts = part_pts[:,2]

        BOXES = np.asarray(box_bag).reshape([-1,3])
        # axl_m = np.amin(BOXES, axis=0)-1
        # axl_M = np.amax(BOXES, axis=0)+1
        # max_range = axl_M - axl_m
        # axl_m = np.min(BOXES)-1
        # axl_M = np.max(BOXES)+1
        axl_m = -1
        axl_M = 48
        max_range = axl_M - axl_m
        ax_1.set_xlim(axl_m,axl_M)
        ax_1.set_ylim(axl_m,axl_M)
        ax_1.set_zlim(axl_m,axl_M)
        ax_1.scatter(X_pts,Y_pts,Z_pts,alpha=0.8)
    elif len(full_mask.shape) == 3:
        # print('Warning: full_mask should be pts format, vox is too costly!')
        axl_m = 0
        axl_M = np.max(full_mask.shape)
        max_range = axl_M - axl_m
        ax_1.set_xlim(axl_m,axl_M)
        ax_1.set_ylim(axl_m,axl_M)
        ax_1.set_zlim(axl_m,axl_M)
        draw_voxel(full_mask, ax_1)  # mask crop

    # draw the bounding boxes of the whole shape
    def drawCubes_roi(Boxes, id_bag, ax_in):
        """ Input Boxes: list of 8*3 np.array, 8 vertices """
        # list of sides' polygons of figure; cube surface
        # print('Boxes num:', Boxes.shape[0])
        for i in range(Boxes.shape[0]):
            minCoord = Boxes[i,0:3]
            maxCoord = Boxes[i,3:6]
            x_m = minCoord[0]
            y_m = minCoord[1]
            z_m = minCoord[2]
            x_M = maxCoord[0]
            y_M = maxCoord[1]
            z_M = maxCoord[2]
            boxes = []
            boxes.append([x_m, y_m, z_m])  # A
            boxes.append([x_M, y_m, z_m])  # B
            boxes.append([x_M, y_m, z_M])  # C
            boxes.append([x_m, y_m, z_M])  # D
            boxes.append([x_m, y_M, z_m])  # E
            boxes.append([x_M, y_M, z_m])  # F
            boxes.append([x_M, y_M, z_M])  # G
            boxes.append([x_m, y_M, z_M])
            Z = boxes
            verts = [[Z[0],Z[1],Z[2],Z[3]],
                     [Z[4],Z[5],Z[6],Z[7]],
                     [Z[0],Z[1],Z[5],Z[4]],
                     [Z[2],Z[3],Z[7],Z[6]],
                     [Z[1],Z[2],Z[6],Z[5]],
                     [Z[4],Z[7],Z[3],Z[0]]]
            # plot sides
            collection = Poly3DCollection(verts,
                linewidths=.25, edgecolors='r', alpha=.15)
            # face_color = [0, 1, 1]
            face_color = COLOR_PALETTE[id_bag[i].astype(int)]
            collection.set_facecolor(face_color)
            ax_in.add_collection3d(collection)

    drawCubes_roi(box_bag, part_ids, ax_1)

    # Create cubic bounding box to simulate equal aspect ratio
    # pdb.set_trace()
    try:
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2].flatten() + 0.5*(axl_M+axl_m)
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2].flatten() + 0.5*(axl_M+axl_m)
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2].flatten() + 0.5*(axl_M+axl_m)
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax_1.plot([xb], [yb], [zb], 'w')
    except UnboundLocalError as e:
        print('Error! ', e)
        pdb.set_trace()


def draw_vox_pts(part_pts):
    """ for visualization, quick """
    # print('drawing mask...')
    plt.close('all')

    fig = plt.figure()
    # plot the mini_mask
    # ax = fig.gca(projection='3d'

    # plot the full size mask: in pts format
    ax_1 = fig.add_subplot(111, projection='3d')
    ax_1.set_title('tfgraph computed box')
    ax_1.set_aspect('equal')
    ax_1.set_xlabel('--X->')
    ax_1.set_ylabel('--Y->')
    ax_1.set_zlabel('--Z->')
    # ax_1.view_init(elev=79., azim=-65.)
    ax_1.view_init(elev=17., azim=165.)
    # pdb.set_trace()

    X_pts = part_pts[:,0]
    Y_pts = part_pts[:,1]
    Z_pts = part_pts[:,2]

    # axl_m = np.amin(BOXES, axis=0)-1
    # axl_M = np.amax(BOXES, axis=0)+1
    # max_range = axl_M - axl_m
    # axl_m = np.min(BOXES)-1
    # axl_M = np.max(BOXES)+1
    axl_m = -1
    axl_M = 16
    ax_1.set_xlim(axl_m,axl_M)
    ax_1.set_ylim(axl_m,axl_M)
    ax_1.set_zlim(axl_m,axl_M)
    ax_1.scatter(X_pts,Y_pts,Z_pts,alpha=0.8)


def draw_volume_shape(voxel_data):
    print('drawing volumes...')
    plt.close('all')

    fig = plt.figure()
    # plot the voxel_data
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('voxel data')
    axl_m = -1
    axl_M = np.max(voxel_data.shape)+1
    ax.set_aspect('equal')
    ax.set_xlim(axl_m,axl_M)
    ax.set_ylim(axl_m,axl_M)
    ax.set_zlim(axl_m,axl_M)
    ax.set_xlabel('--X->')
    ax.set_ylabel('--Y->')
    ax.set_zlabel('--Z->')
    ax.view_init(elev=17., azim=165.)
    draw_voxel(voxel_data, ax)


def draw_volume_seg(vox_seg):
    print('drawing seg volumes...')
    plt.close('all')

    def draw_voxel_v2(voxel_data, ax):
        def explode(data):
            # explode voxel data
            size = np.array(data.shape)*2
            # insert gaps between valid grid
            data_e = np.zeros(size - 1, dtype=data.dtype)
            data_e[::2, ::2, ::2] = data
            return data_e

        # build up the color voxel grid
        h,w,d = voxel_data.shape
        colors = np.empty((h,w,d), dtype=object)  # (r,g,b,a) at each position

        colors[voxel_data==0] = '#7A88CC00'
        colors[voxel_data==1] = 'red'
        colors[voxel_data==2] = 'blue'
        colors[voxel_data==3] = 'green'
        colors[voxel_data==4] = 'violet'
        colors[voxel_data==5] = 'cyan'
        colors[voxel_data==6] = 'lightsalmon'
        colors[voxel_data==7] = 'gray'

        facecolors = colors
        # facecolors = np.where(voxel_data, '#FFD65DC0', '#7A88CC00')  # '#7A88CCC0'
        edgecolors = np.where(voxel_data, '#BFAB6E', '#7D84A600')  # '#7D84A6'
        filled = np.ones(voxel_data.shape)

        # upscale the above voxel image, leaving gaps
        filled_2 = explode(filled)
        fcolors_2 = explode(facecolors)
        ecolors_2 = explode(edgecolors)
        # pdb.set_trace()

        # Shrink the gaps, gap size = 0.1
        x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
        x[0::2, :, :] += 0.05
        y[:, 0::2, :] += 0.05
        z[:, :, 0::2] += 0.05
        x[1::2, :, :] += 0.95
        y[:, 1::2, :] += 0.95
        z[:, :, 1::2] += 0.95

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=fcolors_2)

    fig = plt.figure()
    # plot the vox_seg
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('seg volumes')
    axl_m = -1
    axl_M = np.max(vox_seg.shape)+1
    ax.set_aspect('equal')
    ax.set_xlim(axl_m,axl_M)
    ax.set_ylim(axl_m,axl_M)
    ax.set_zlim(axl_m,axl_M)
    ax.set_xlabel('--X->')
    ax.set_ylabel('--Y->')
    ax.set_zlabel('--Z->')
    ax.view_init(elev=17., azim=165.)
    draw_voxel_v2(vox_seg, ax)


def load_h5_mask(h5_filename):
    f = h5py.File(h5_filename, 'r')
    bbox = f['bbox'][:]
    mask = f['mask'][:]
    label = f['label'][:]
    return (mask, bbox, label)


def get_mask_data(vox_datas, pred_segs, vox_segs, Mini_Shape, Max_Ins_Num):
    """ input a batch of voxels and gt_segs, pred_segs
        compute boxes (rois factors)
        output: proposals (crop from voxels)
                mask targets (crop from segs)
    """
    batch_size = vox_datas.shape[0]
    Bbox_list = []
    Mask_list = []
    Prop_list = []
    Pid_list = []
    for b in range(batch_size):
        cur_vox = np.squeeze(vox_datas[b, ...])
        cur_pred_seg = np.squeeze(pred_segs[b, ...])
        cur_gt_seg = np.squeeze(vox_segs[b, ...])
        pts = np.transpose(np.array(np.where(cur_vox)))
        plb = cur_pred_seg[pts[:,0], pts[:,1], pts[:,2]]
        pts = pts.astype(float)

        # pdb.set_trace()
        """ process seg pts into subpart groups (distance metric, group box)"""
        # tic()
        # with warnings.catch_warnings():
        #     warnings.filterwarnings('error')
        #     try:
        #         Boxes, BLabels, _, _ = Box.computeBox(pts=pts, plb=plb, alpha=1.5)
        #     except RuntimeWarning as e:
        #         print('Error found:', e)
        #         pdb.set_trace()
        Boxes, BLabels, _, _ = Box.computeBox(pts=pts, plb=plb, alpha=1.5)
        # toc()  # computeBox(): 0.5~3 seconds per model
        # ### Todo: augment Boxes data
        # tic()
        gt_masks, proposals, boxes, part_ids = gen_masks_info_v3(
                cur_gt_seg, Boxes, BLabels, mask_shape=Mini_Shape)  # mini_masks: gt mask
        """ gt_masks: (subpart_num, mini_shape**3)
            proposals: (subpart_num, mini_shape**3)
            part_ids: (subpart_num,)
        """
        # toc()  # gen_masks_info_v3 spend too much time, almost 4~7 times of computeBox
        Bbox_data = np.zeros((Max_Ins_Num, 6))  # [rois, x1,y1,z1, x2,y2,z2]
        Mask_data = np.zeros((Max_Ins_Num, Mini_Shape, Mini_Shape, Mini_Shape))
        Prop_data = np.zeros((Max_Ins_Num, Mini_Shape, Mini_Shape, Mini_Shape))
        Label_data = np.zeros((Max_Ins_Num,))
        b_count = len(part_ids)  # box count
        if b_count <= Max_Ins_Num:
            Bbox_data[0:b_count, ...] = np.asarray(boxes)
            Mask_data[0:b_count, ...] = np.asarray(gt_masks)
            Prop_data[0:b_count, ...] = np.asarray(proposals)
            Label_data[0:b_count, ...] = np.asarray(part_ids)
        elif b_count > Max_Ins_Num:
            print("Warning: box count larger than Max_Ins_Num! ", b_count, '/', Max_Ins_Num)
            b_count = Max_Ins_Num
            Bbox_data[0:b_count, ...] = np.asarray(boxes)[0:b_count, ...]
            Mask_data[0:b_count, ...] = np.asarray(gt_masks)[0:b_count, ...]
            Prop_data[0:b_count, ...] = np.asarray(proposals)[0:b_count, ...]
            Label_data[0:b_count, ...] = np.asarray(part_ids)[0:b_count, ...]
        Bbox_list.append(np.expand_dims(Bbox_data, axis=0))
        Mask_list.append(np.expand_dims(Mask_data, axis=0))
        Prop_list.append(np.expand_dims(Prop_data, axis=0))
        Pid_list.append(np.expand_dims(Label_data, axis=0))

    return np.concatenate(Prop_list, 0), np.concatenate(Mask_list, 0),\
        np.concatenate(Pid_list, 0), np.concatenate(Bbox_list, 0)


# if __name__ == '__main__':
#     """ Settings """
#     DATA_DIR = os.path.join(BASE_DIR, 'hdf5_data'+g_.Data_suf, '48')
#     # ['Motorbike', 'Earphone', 'Rocket', 'Airplane', 'Chair']
#     CAT_LIST = ['Motorbike']
#     dataType = 'vol'  # 'pts'
#     train_or_test_list = ['test', 'train', 'val']  # ['test', 'train', 'val']

#     debug = False
#     view_flag = True
#     write_flag = False
#     Max_Ins_Num = g_.MAX_INS_NUM  # 15
#     Mini_Shape = g_.MASK_SHAPE  # 15
#     for CAT_NAME in CAT_LIST:
#         for train_or_test in train_or_test_list:
#             print('Category: ', CAT_NAME, '\t', train_or_test)
#             out_dir = os.path.join(DATA_DIR, 'mask')
#             if not os.path.exists(out_dir):
#                 os.mkdir(out_dir)
#             dump_dir = './cubeImgs/'+CAT_NAME+g_.Data_suf
#             if not os.path.exists(dump_dir):
#                 os.mkdir(dump_dir)

#             m_file = os.path.join(DATA_DIR, CAT_NAME+'_'+dataType+'_'+train_or_test+'.h5')
#             model_num = Box.get_data_num(m_file)

#             current_data, current_seg = Box.load_h5_volumes_data(m_file)
#             input_cur_data = current_data[2:3, ...]
#             input_cur_seg = current_seg[2:3, ...]
#             pred_seg = input_cur_seg
#             cur_proposals, cur_targets, cur_partids, _= get_mask_data(
#                     input_cur_data, pred_seg, input_cur_seg,
#                     Mini_Shape=g_.MASK_SHAPE, Max_Ins_Num=g_.MAX_INS_NUM)

#             if view_flag and (train_or_test=='test'):
#                 mini_mask = cur_proposals[0,2,...]
#                 part_id = cur_partids[0,2]
#                 # draw_mask(mini_mask, crop_mask)
#                 # draw_mask_v2(mini_mask, mask_pts, box_bag=Boxes, part_ids=part_ids)
#                 draw_volume_shape(mini_mask)
#                 # figname = os.path.join(dump_dir, 'model_'+str(m_idx)+
#                 #           '-part_'+str(part_id)+'-box_'+str(i)+'.png')
#                 # plt.savefig(figname)
#                 pdb.set_trace()


""" __main__ """
if __name__ == '__main__':
    """ Settings """
    DATA_DIR = os.path.join(BASE_DIR, 'hdf5_data'+g_.Data_suf, '48')
    # ['Motorbike', 'Earphone', 'Rocket', 'Airplane', 'Chair']
    CAT_LIST = ['Motorbike', 'Earphone', 'Airplane']
    dataType = 'vol'  # 'pts'
    train_or_test_list = ['test', 'train']  # ['test', 'train', 'val']

    debug = False
    view_flag = True
    write_flag = True
    Max_Ins_Num = g_.MAX_INS_NUM  # 30
    Mini_Shape = g_.MASK_SHAPE  # 16
    for CAT_NAME in CAT_LIST:
        for train_or_test in train_or_test_list:
            print('Category: ', CAT_NAME, '\t', train_or_test)
            out_dir = os.path.join(DATA_DIR, 'bbox')
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            dump_dir = './bbox_gt/'+CAT_NAME+g_.Data_suf
            if not os.path.exists(dump_dir):
                os.mkdir(dump_dir)

            m_file = os.path.join(DATA_DIR, CAT_NAME+'_'+dataType+'_'+train_or_test+'.h5')
            model_num = Box.get_data_num(m_file)

            if write_flag:
                h5_fname = os.path.join(out_dir, CAT_NAME+'_'+dataType+'_'+train_or_test+'.h5')
                h5_fout = h5py.File(h5_fname, 'a')

            for m_idx in range(model_num):
                if debug:
                    m_idx = 8
                print('-----proccessing model in ', train_or_test, ' set: ', m_idx+1, ' of ', model_num, '-----')
                if dataType == 'pts':
                    current_pts, current_plb = Box.load_h5_pts_data(m_file)
                    pts, indices = np.unique(current_pts[m_idx], return_index=True, axis=0)
                    plb = current_plb[m_idx][indices]
                elif dataType == 'vol':
                    vox_data, vox_seg = Box.load_h5_volumes_data(m_file)
                    cur_vox = np.squeeze(vox_data[m_idx, ...])
                    cur_seg = np.squeeze(vox_seg[m_idx, ...])
                    pts = np.transpose(np.array(np.where(cur_vox)))
                    plb = cur_seg[pts[:,0], pts[:,1], pts[:,2]]
                    pts = pts.astype(float)

                # pdb.set_trace()
                """ process seg pts into subpart groups (distance metric, group box)"""
                tic()
                Boxes, BLabels, parts_bag, label_bag = Box.computeBox(pts=pts, plb=plb, alpha=1.5)
                boxes, part_ids = Box.gen_rois_info(Boxes, BLabels)
                toc()  # computeBox(): 0.5 seconds per model

                if write_flag:
                    """ write to h5 file
                        bbox, mini_mask, box_label
                    """
                    # unify the data to be saved
                    Bbox_data = np.zeros((Max_Ins_Num, 6))  # [rois, x1,y1,z1, x2,y2,z2]
                    Label_data = np.zeros((Max_Ins_Num,))
                    b_count = len(part_ids)  # box count
                    if b_count <= Max_Ins_Num:
                        Bbox_data[0:b_count, ...] = np.asarray(boxes)
                        Label_data[0:b_count, ...] = np.asarray(part_ids)
                    elif b_count > Max_Ins_Num:
                        print("Warning: box count larger than Max_Ins_Num!")
                        b_count = Max_Ins_Num
                        Bbox_data[0:b_count, ...] = np.asarray(boxes)[0:b_count, ...]
                        Label_data[0:b_count, ...] = np.asarray(part_ids)[0:b_count, ...]

                    if m_idx == 0:
                        bbox_set = h5_fout.create_dataset('bbox', (1, Max_Ins_Num, 6),
                                   maxshape=(None, Max_Ins_Num, 6))
                        label_set = h5_fout.create_dataset('label', (1, Max_Ins_Num),
                                    maxshape=(None, Max_Ins_Num))
                        bbox_set[:] = np.expand_dims(Bbox_data, axis=0)
                        label_set[:] = np.expand_dims(Label_data, axis=0)
                    else:  # resize the 0 axis
                        bbox_set = h5_fout['bbox']
                        label_set = h5_fout['label']

                        bbox_set.resize(bbox_set.shape[0]+1, axis=0)
                        label_set.resize(label_set.shape[0]+1, axis=0)

                        bbox_set[-1:, ...] = np.expand_dims(Bbox_data, axis=0)
                        label_set[-1:, ...] = np.expand_dims(Label_data, axis=0)

                    h5_fout.flush()

                """ visualization """
                if view_flag and (train_or_test=='test'):
                    bboxes = np.asarray(boxes)
                    partids = part_ids
                    part_list = []
                    for i in range(bboxes.shape[0]):
                        part_id = partids[i]
                        if part_id == 0 or part_id in part_list:
                            continue

                        pts_idx = plb == part_id
                        cur_partpts = pts[pts_idx,:]
                        draw_mask_box(cur_partpts, bboxes, partids)
                        figname = os.path.join(dump_dir, 'model_'+train_or_test+'_'+str(m_idx)+
                                  '-part_'+str(part_id)+'-box_num'+str(bboxes.shape[0])+'.png')
                        part_list.append(part_id)
                        plt.savefig(figname)

                        # pdb.set_trace()

                if debug:
                    pdb.set_trace()
                    # compare mini_mask and expand_mask

            if write_flag:
                h5_fout.close()


# if __name__ == '__main__':
#     """ Settings """
#     DATA_DIR = os.path.join(BASE_DIR, 'hdf5_data'+g_.Data_suf, '48')
#     # ['Motorbike', 'Earphone', 'Rocket', 'Airplane', 'Chair']
#     CAT_LIST = ['Airplane']
#     dataType = 'vol'  # 'pts'
#     train_or_test_list = ['test', 'train', 'val']  # ['test', 'train', 'val']

#     debug = False
#     view_flag = False
#     write_flag = False
#     Max_Ins_Num = g_.MAX_INS_NUM  # 15
#     Mini_Shape = g_.MASK_SHAPE  # 15
#     for CAT_NAME in CAT_LIST:
#         for train_or_test in train_or_test_list:
#             print('Category: ', CAT_NAME, '\t', train_or_test)
#             out_dir = os.path.join(DATA_DIR, 'mask')
#             if not os.path.exists(out_dir):
#                 os.mkdir(out_dir)
#             dump_dir = './cubeImgs/'+CAT_NAME+g_.Data_suf
#             if not os.path.exists(dump_dir):
#                 os.mkdir(dump_dir)

#             m_file = os.path.join(DATA_DIR, CAT_NAME+'_'+dataType+'_'+train_or_test+'.h5')
#             model_num = Box.get_data_num(m_file)

#             if write_flag:
#                 h5_fname = os.path.join(out_dir, CAT_NAME+'_'+dataType+'_'+train_or_test+'.h5')
#                 h5_fout = h5py.File(h5_fname, 'a')

#             for m_idx in range(model_num):
#                 if debug:
#                     m_idx = 8
#                 print('-----proccessing model in ', train_or_test, ' set: ', m_idx+1, ' of ', model_num, '-----')
#                 if dataType == 'pts':
#                     current_pts, current_plb = Box.load_h5_pts_data(m_file)
#                     pts, indices = np.unique(current_pts[m_idx], return_index=True, axis=0)
#                     plb = current_plb[m_idx][indices]
#                 elif dataType == 'vol':
#                     vox_data, vox_seg = Box.load_h5_volumes_data(m_file)
#                     cur_vox = np.squeeze(vox_data[m_idx, ...])
#                     cur_seg = np.squeeze(vox_seg[m_idx, ...])
#                     pts = np.transpose(np.array(np.where(cur_vox)))
#                     plb = cur_seg[pts[:,0], pts[:,1], pts[:,2]]
#                     pts = pts.astype(float)

#                 # pdb.set_trace()
#                 """ process seg pts into subpart groups (distance metric, group box)"""
#                 tic()
#                 Boxes, BLabels, parts_bag, label_bag = Box.computeBox(pts=pts, plb=plb, alpha=1.5)
#                 toc()  # computeBox(): 0.5 seconds per model
#                 crop_masks, mini_masks, boxes, part_ids = gen_masks_info_v2(
#                         cur_seg, Boxes, BLabels, mask_shape=Mini_Shape)
#                 part_pts = gen_masks_info_v1(parts_bag, label_bag, mask_shape=5)

#                 if write_flag:
#                     """ write to h5 file
#                         bbox, mini_mask, box_label
#                     """
#                     # unify the data to be saved
#                     Bbox_data = np.zeros((Max_Ins_Num, 6))  # [rois, x1,y1,z1, x2,y2,z2]
#                     Mask_data = np.zeros((Max_Ins_Num, Mini_Shape, Mini_Shape, Mini_Shape))
#                     Label_data = np.zeros((Max_Ins_Num,))
#                     b_count = len(part_ids)  # box count
#                     if b_count <= Max_Ins_Num:
#                         Bbox_data[0:b_count, ...] = np.asarray(boxes)
#                         Mask_data[0:b_count, ...] = np.asarray(mini_masks)
#                         Label_data[0:b_count, ...] = np.asarray(part_ids)
#                     elif b_count > Max_Ins_Num:
#                         print("Warning: box count larger than Max_Ins_Num!")
#                         b_count = Max_Ins_Num
#                         Bbox_data[0:b_count, ...] = np.asarray(boxes)[0:b_count, ...]
#                         Mask_data[0:b_count, ...] = np.asarray(mini_masks)[0:b_count, ...]
#                         Label_data[0:b_count, ...] = np.asarray(part_ids)[0:b_count, ...]

#                     if m_idx == 0:
#                         bbox_set = h5_fout.create_dataset('bbox', (1, Max_Ins_Num, 6),
#                                    maxshape=(None, Max_Ins_Num, 6))
#                         mask_set = h5_fout.create_dataset('mask',
#                                    (1, Max_Ins_Num, Mini_Shape, Mini_Shape, Mini_Shape),
#                                    maxshape=(None, Max_Ins_Num, Mini_Shape, Mini_Shape, Mini_Shape))
#                         label_set = h5_fout.create_dataset('label', (1, Max_Ins_Num),
#                                     maxshape=(None, Max_Ins_Num))
#                         bbox_set[:] = np.expand_dims(Bbox_data, axis=0)
#                         mask_set[:] = np.expand_dims(Mask_data, axis=0)
#                         label_set[:] = np.expand_dims(Label_data, axis=0)
#                     else:  # resize the 0 axis
#                         bbox_set = h5_fout['bbox']
#                         mask_set = h5_fout['mask']
#                         label_set = h5_fout['label']

#                         bbox_set.resize(bbox_set.shape[0]+1, axis=0)
#                         mask_set.resize(mask_set.shape[0]+1, axis=0)
#                         label_set.resize(label_set.shape[0]+1, axis=0)

#                         bbox_set[-1:, ...] = np.expand_dims(Bbox_data, axis=0)
#                         mask_set[-1:, ...] = np.expand_dims(Mask_data, axis=0)
#                         label_set[-1:, ...] = np.expand_dims(Label_data, axis=0)

#                     h5_fout.flush()

#                 """ visualization """
#                 if view_flag and (train_or_test=='test'):
#                     for i in range(len(part_ids)):
#                         crop_mask = crop_masks[i]
#                         mask_pts = part_pts[i]
#                         mini_mask = mini_masks[i]
#                         part_id = part_ids[i]
#                         # draw_mask(mini_mask, crop_mask)
#                         draw_mask_v2(mini_mask, mask_pts, box_bag=Boxes, part_ids=part_ids)
#                         figname = os.path.join(dump_dir, 'model_'+str(m_idx)+
#                                   '-part_'+str(part_id)+'-box_'+str(i)+'.png')
#                         plt.savefig(figname)
#                         # pdb.set_trace()

#                 if debug:
#                     draw_mask(mini_masks[3], crop_masks[3])
#                     pdb.set_trace()
#                     # compare mini_mask and expand_mask
#                     ip = 3
#                     _, m_excrop = expand_mask(boxes[ip], mini_masks[ip], vox_shape=48)
#                     draw_mask(m_excrop, crop_masks[ip])
#                     pdb.set_trace()

#             if write_flag:
#                 h5_fout.close()
