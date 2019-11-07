""" some functions used by network training & testing
    Written by W.Zj
"""
""" 2018.03.27, Tue """

import numpy as np
from scipy.interpolate import griddata
import tensorflow as tf
import itertools
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb


def compute_overlaps_masks(masks1, masks2):
    '''Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, Depth, instances]
    Return: [pred_masks, gt_masks], like a confusion matrix
    '''
    # flatten masks
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = (intersections + 1e-8)/ (union + 1e-8)

    return overlaps


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
    mini_mask = griddata(ijk, maskAux, (i,j,k), method="linear")  # >= 0.5

    return mini_mask


def expand_mask(bbox, mini_mask, vox_shape):
    """Resizes mini masks back to voxel size. Reverses the change
    of minimize_mask().
    return mask : [vox_shape, vox_shape, vox_shape], [0,1] float value
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
    mask_cropped = griddata(ijk, mini_maskAux, (i,j,k), method="linear")  # >= 0.5

    mask = np.zeros([vox_shape, vox_shape, vox_shape])
    mask[x_m:(x_M+1), y_m:(y_M+1), z_m:(z_M+1)] = mask_cropped

    return mask


def plot_overlaps(gt_class_ids, pred_class_ids, pred_scores,
                  overlaps, class_names, savename, threshold=0.5):
    """Draw a grid showing how ground truth objects are classified.
    gt_class_ids: [N] int. Ground truth class IDs
    pred_class_id: [N] int. Predicted class IDs
    pred_scores: [N] float. The probability scores of predicted classes
    overlaps: [pred_boxes, gt_boxes] IoU overlaps of predictins and GT boxes.
    class_names: list of all class names in the dataset
    threshold: Float. The prediction probability required to predict a class
    """
    # pdb.set_trace()
    plt.close('all')
    # gt_class_ids = gt_class_ids[gt_class_ids != 0]
    # pred_class_ids = pred_class_ids[pred_class_ids != 0]

    plt.figure(figsize=(12, 10))
    plt.imshow(overlaps, interpolation='nearest', cmap=plt.cm.Blues)
    try:
        plt.yticks(np.arange(len(pred_class_ids)),
                   ["{} ({:.2f})".format(class_names[int(id)], pred_scores[i])
                    for i, id in enumerate(pred_class_ids)])
        plt.xticks(np.arange(len(gt_class_ids)),
                   [class_names[int(id)] for id in gt_class_ids], rotation=90)
    except IndexError as e:
        print(e)
        pdb.set_trace()
    thresh = overlaps.max() / 2.
    for i, j in itertools.product(range(overlaps.shape[0]),
                                  range(overlaps.shape[1])):
        text = ""
        if overlaps[i, j] > threshold:
            text = "match" if gt_class_ids[j] == pred_class_ids[i] else "wrong"
        color = ("white" if overlaps[i, j] > thresh
                 else "black" if overlaps[i, j] > 0
                 else "grey")
        plt.text(j, i, "{:.3f}\n{}".format(overlaps[i, j], text),
                 horizontalalignment="center", verticalalignment="center",
                 fontsize=9, color=color)

    plt.tight_layout()
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.savefig(savename)


# for rpn graph
def overlaps_bbox(boxes1, boxes2):
    """Numpy function
    Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    Return: overlaps, [boxes1.shape[0],boxes2.shape[0]]
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    b1 = np.repeat(boxes1, boxes2.shape[0], axis=0)
    b2 = np.tile(boxes2, [boxes1.shape[0], 1])
    # 2. Compute intersections
    b1_x1, b1_y1, b1_z1, b1_x2, b1_y2, b1_z2 = np.split(b1, 6, axis=1)
    b2_x1, b2_y1, b2_z1, b2_x2, b2_y2, b2_z2 = np.split(b2, 6, axis=1)
    z1 = np.maximum(b1_z1, b2_z1)
    y1 = np.maximum(b1_y1, b2_y1)
    x1 = np.maximum(b1_x1, b2_x1)
    z2 = np.minimum(b1_z2, b2_z2)
    y2 = np.minimum(b1_y2, b2_y2)
    x2 = np.minimum(b1_x2, b2_x2)
    intersection = np.maximum(x2-x1, 0) * np.maximum(y2-y1, 0) * np.maximum(z2-z1, 0)
    # 3. Compute unions
    b1_volume = (b1_z2 - b1_z1) * (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_volume = (b2_z2 - b2_z1) * (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_volume + b2_volume - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = (intersection) / (union + 1e-8)
    overlaps = np.reshape(iou, [boxes1.shape[0], boxes2.shape[0]])
    return overlaps


def box_refinement_delta(box, gt_box):
    """Numpy function
    Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (x1, y1, z1, x2, y2, z2)]
    in other words, get deltas
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    length = box[:,3] - box[:,0]
    width = box[:,4] - box[:,1]
    height = box[:,5] - box[:,2]
    center_x = 0.5*(box[:,0] + box[:,3])
    center_y = 0.5*(box[:,1] + box[:,4])
    center_z = 0.5*(box[:,2] + box[:,5])

    gt_length = gt_box[:,3] - gt_box[:,0]
    gt_width = gt_box[:,4] - gt_box[:,1]
    gt_height = gt_box[:,5] - gt_box[:,2]
    gt_center_x = 0.5*(gt_box[:,0] + gt_box[:,3])
    gt_center_y = 0.5*(gt_box[:,1] + gt_box[:,4])
    gt_center_z = 0.5*(gt_box[:,2] + gt_box[:,5])

    # length[length==0] = 1e-8
    # width[width==0] = 1e-8
    # height[height==0] = 1e-8
    dx = (gt_center_x - center_x) / length
    dy = (gt_center_y - center_y) / width
    dz = (gt_center_z - center_z) / height
    dl = np.log(gt_length/length)
    dw = np.log(gt_width/width)
    dh = np.log(gt_height/height)

    result = np.stack([dx,dy,dz,dl,dw,dh], axis=1)
    return result


def get_rpn_bbox_target(input_rois, part_ids, gt_boxes, gt_partids, threshold=0.0):
    """ Numpy function, need run sess to get input_rois info first

        input_rois: [B,num_rois,(x1,y1,z1,x2,y2,z2)], as anchors
        Return: Not Normalized!
            the target_gt_rois corresponding to rois;
            rois_purified (remove noise rois)
            part_ids of rois_purified
    """
    batch_size = input_rois.shape[0]
    rois_purified = []
    partids_purified = []
    target_rois = []
    target_partids = []
    target_deltas = []
    # for each batch
    for i in range(batch_size):
        cur_rois = input_rois[i]
        cur_part_ids = part_ids[i]
        cur_gt_rois = gt_boxes[i]
        cur_gt_partids = gt_partids[i]

        rois_valid = np.zeros(cur_rois.shape)
        partids_valid = np.zeros(cur_part_ids.shape)
        gt_rois_valid = np.zeros(cur_gt_rois.shape)
        gt_partids_valid = np.zeros(cur_gt_partids.shape)
        target_deltas_valid = np.zeros(cur_gt_rois.shape)

        non_zeros = cur_part_ids != 0
        non_zeros_gt = cur_gt_partids != 0
        cur_rois_ = cur_rois[non_zeros,...]
        cur_part_ids_ = cur_part_ids[non_zeros,...]
        cur_gt_rois_ = cur_gt_rois[non_zeros_gt,...]
        cur_gt_partids_ = cur_gt_partids[non_zeros_gt,...]
        # pdb.set_trace()
        bbox_ious = overlaps_bbox(cur_rois_, cur_gt_rois_)
        # using Unet pred part labels
        part_mask = []
        for ii in range(len(cur_part_ids_)):
            pid = cur_part_ids_[ii]
            part_mask.append(cur_gt_partids_ == pid)
        part_mask = np.array(part_mask, dtype=np.float32)

        roi_iou_max = np.amax(bbox_ious*part_mask, axis=1)
        gt_roi_idx = np.argmax(bbox_ious*part_mask, axis=1)
        valid_bool = roi_iou_max > threshold
        valid_num = np.sum(valid_bool, dtype=np.int)
        valid_gt_idx = gt_roi_idx[valid_bool,...]

        rois_valid[0:valid_num,...] = cur_rois_[valid_bool, ...]
        partids_valid[0:valid_num,...] = cur_part_ids_[valid_bool, ...]
        gt_rois_valid[0:valid_num,...] = cur_gt_rois_[valid_gt_idx, ...]
        gt_partids_valid[0:valid_num,...] = cur_gt_partids_[valid_gt_idx, ...]

        targets_delta = box_refinement_delta(cur_rois_[valid_bool, ...],
                cur_gt_rois_[valid_gt_idx, ...])
        target_deltas_valid[0:valid_num, ...] = targets_delta

        rois_purified.append(np.expand_dims(rois_valid,axis=0))
        partids_purified.append(np.expand_dims(partids_valid,axis=0))
        target_rois.append(np.expand_dims(gt_rois_valid,axis=0))
        target_partids.append(np.expand_dims(gt_partids_valid,axis=0))
        target_deltas.append(np.expand_dims(target_deltas_valid,axis=0))
    # pdb.set_trace()
    return (np.concatenate(rois_purified, axis=0),
            np.concatenate(partids_purified, axis=0),
            np.concatenate(target_deltas, axis=0),
            np.concatenate(target_partids, axis=0))


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [B,N, 6] where each row is x1,y1,z1, x2, y2, z2
    deltas: [B,N, 6] where each row is [dx, dy, dz, log(dl), log(dw), log(dh)]
    """
    assert len(boxes.shape)==3, "input should be batch"
    result = []
    for i in range(boxes.shape[0]):
        cur_boxes = boxes[i, ...]
        cur_deltas = deltas[i, ...]
        length = cur_boxes[:,3] - cur_boxes[:,0]
        width = cur_boxes[:,4] - cur_boxes[:,1]
        height = cur_boxes[:,5] - cur_boxes[:,2]
        center_x = 0.5*(cur_boxes[:,0] + cur_boxes[:,3])
        center_y = 0.5*(cur_boxes[:,1] + cur_boxes[:,4])
        center_z = 0.5*(cur_boxes[:,2] + cur_boxes[:,5])
        # Apply deltas
        center_x += cur_deltas[:,0]*length
        center_y += cur_deltas[:,1]*width
        center_z += cur_deltas[:,2]*height
        length *= np.exp(cur_deltas[:,3])
        width *= np.exp(cur_deltas[:,4])
        height *= np.exp(cur_deltas[:,5])
        # Convert to x1,y1,z1,x2,y2,z2
        x1 = center_x - 0.5*length
        y1 = center_y - 0.5*width
        z1 = center_z - 0.5*height
        x2 = x1 + length
        y2 = y1 + width
        z2 = z1 + height

        cur_result = np.stack([x1,y1,z1,x2,y2,z2], axis=1)
        result.append(np.expand_dims(cur_result, axis=0))
    results = np.concatenate(result, axis=0)
    return results


###########################
# TF graph version, for rpn graph
###########################
# def detect_rpn_targets_graph(input_rois, part_ids, gt_boxes, gt_partids):
#     batch_size = input_rois.get_shape()[0]
#     rois_purified = []
#     partids_purified = []
#     target_rois = []
#     target_partids = []
#     # for each batch
#     for i in range(batch_size):
#         cur_rois = tf.squeeze(input_rois[i])
#         cur_part_ids = tf.squeeze(part_ids[i])
#         cur_gt_rois = tf.squeeze(gt_boxes[i])
#         cur_gt_partids = tf.squeeze(gt_partids[i])

#         cur_rois_, non_zeros = trim_zeros_graph(cur_rois)
#         cur_gt_rois_, non_zeros_gt = trim_zeros_graph(cur_gt_rois)
#         cur_part_ids_ = tf.boolean_mask(cur_part_ids, non_zeros)
#         cur_gt_partids_ = tf.boolean_mask(cur_gt_partids, non_zeros_gt)

#         rois_valid = np.zeros(cur_rois.shape)
#         partids_valid = np.zeros(cur_part_ids.shape)
#         gt_rois_valid = np.zeros(cur_gt_rois.shape)
#         gt_partids_valid = np.zeros(cur_gt_partids.shape)

#         non_zeros = cur_part_ids != 0
#         non_zeros_gt = cur_gt_partids != 0
#         cur_rois_ = cur_rois[non_zeros,...]
#         cur_part_ids_ = cur_part_ids[non_zeros,...]
#         cur_gt_rois_ = cur_gt_rois[non_zeros_gt,...]
#         cur_gt_partids_ = cur_gt_partids[non_zeros_gt,...]
#         # pdb.set_trace()
#         bbox_ious = overlaps_bbox(cur_rois_, cur_gt_rois_)
#         # using Unet pred part labels
#         part_mask = []
#         for ii in range(len(cur_part_ids_)):
#             pid = cur_part_ids_[ii]
#             part_mask.append(cur_gt_partids_ == pid)
#         part_mask = np.array(part_mask, dtype=np.float32)

#         roi_iou_max = np.amax(bbox_ious*part_mask, axis=1)
#         gt_roi_idx = np.argmax(bbox_ious*part_mask, axis=1)
#         valid_bool = roi_iou_max >= 0.005
#         valid_num = np.sum(valid_bool, dtype=np.int)
#         valid_gt_idx = gt_roi_idx[valid_bool,...]

#         rois_valid[0:valid_num,...] = cur_rois_[valid_bool, ...]
#         partids_valid[0:valid_num,...] = cur_part_ids_[valid_bool, ...]
#         gt_rois_valid[0:valid_num,...] = cur_gt_rois_[valid_gt_idx, ...]
#         gt_partids_valid[0:valid_num,...] = cur_gt_partids_[valid_gt_idx, ...]

#     rois_purified.append(np.expand_dims(rois_valid,axis=0))
#     partids_purified.append(np.expand_dims(partids_valid,axis=0))
#     target_rois.append(np.expand_dims(gt_rois_valid,axis=0))
#     target_partids.append(np.expand_dims(gt_partids_valid,axis=0))

#     return (np.concatenate(rois_purified, axis=0),
#             np.concatenate(partids_purified, axis=0),
#             np.concatenate(target_rois, axis=0),
#             np.concatenate(target_partids, axis=0))


def trim_zeros_graph(boxes, name=None):
    """Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros
