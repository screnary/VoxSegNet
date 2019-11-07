import numpy as np
import my_util
import pdb


def assess_Unet_result(pred_val, current_seg, loss_Unet_val, part_num, acc_ops):
    """ evaluate pred (Unet) quality while network running """

    # iou_oids = range(part_num)
    iou_oids = range(1, part_num+1, 1)
    BATCH_SIZE = pred_val.shape[0]
    VOX_SIZE = pred_val.shape[1]
    pred_val = np.reshape(np.argmax(pred_val, -1), (BATCH_SIZE, VOX_SIZE, VOX_SIZE, VOX_SIZE))

    cur_seg_new = current_seg
    # 0 is background label! # pred is from 0 value, but seg gt is from 1 value (0 for the back)
    mask = cur_seg_new != 0
    mask = mask.astype(np.float32)
    correct = np.sum((pred_val == cur_seg_new)*mask, axis=(1,2,3))
    seen_per_instance = np.sum(mask, axis=(1,2,3))
    acc_per_instance = np.array(correct) / np.array(seen_per_instance)

    acc_ops['total_accuracy'] += np.sum(acc_per_instance)
    acc_ops['total_seen'] += BATCH_SIZE
    acc_ops['loss_Unet_sum'] += loss_Unet_val

    iou_log = ''  # iou details string
    intersect_mask = np.int32((pred_val == cur_seg_new)*mask)  # [B,V,V,V]
    # pdb.set_trace()
    for bid in range(BATCH_SIZE):
        # bid # batch id
        total_iou = 0.0  # for this 3D shape.
        intersect_mask_bid = intersect_mask[bid, ...]
        mask_bid = mask[bid, ...]
        pred_val_bid = pred_val[bid, ...]
        cur_seg_bid = cur_seg_new[bid, ...]
        for oid in iou_oids:
            n_pred = np.sum((pred_val_bid == oid) * mask_bid)  # only the valid grids' pred
            # n_pred = np.sum(seg_pred_val == oid)
            n_gt = np.sum(cur_seg_bid == oid)
            n_intersect = np.sum(np.int32(cur_seg_bid == oid) * intersect_mask_bid)
            n_union = n_pred + n_gt - n_intersect
            iou_log += '_pred:' + str(n_pred) + '_gt:' + str(n_gt) + '_intersect:' + str(n_intersect) + '_union:' + str(n_union) + '_'
            if n_union == 0:
                total_iou += 1
                iou_log += '_:1\n'
            else:
                total_iou += n_intersect * 1.0 / n_union  # sum across parts
                iou_log += '_:'+str(n_intersect*1.0/n_union)+'\n'

        avg_iou = total_iou / len(iou_oids)  # average iou across parts, for one object
        # pdb.set_trace()
        acc_ops['total_acc_iou'] += avg_iou

    return avg_iou, acc_per_instance, iou_log


def assess_mask_assign_result(new_result, input_seg, part_num, acc_ops):
    """ input should have shape [batch, vox,vox,vox]"""
    # iou_oids = range(part_num)
    iou_oids = range(1, part_num+1, 1)
    if len(new_result.shape) < 4:
        new_result = np.expand_dims(new_result, axis=0)
    pred_val = new_result

    # cur_seg_new = input_seg - 1.0  # pred is from 0 value, but seg gt is from 1 value (0 for the back)
    # mask = cur_seg_new != -1
    cur_seg_new = input_seg  # pred is from 0 value, but seg gt is from 1 value (0 for the back)
    mask = cur_seg_new != 0
    mask = mask.astype(np.float32)
    correct = np.sum((pred_val == cur_seg_new)*mask, axis=(1,2,3))
    seen_per_instance = np.sum(mask, axis=(1,2,3))
    acc_per_instance = np.array(correct) / np.array(seen_per_instance)
    acc_ops['total_mask_assign_accuracy'] += np.sum(acc_per_instance)
    # pdb.set_trace()
    iou_log = ''  # iou details string
    intersect_mask = np.int32((pred_val == cur_seg_new)*mask)  # [B,V,V,V]
    # pdb.set_trace()
    for bid in range(1):  # range(BATCH_SIZE)
        # bid # batch id
        total_iou = 0.0  # for this 3D shape.
        intersect_mask_bid = intersect_mask[bid, ...]
        mask_bid = mask[bid, ...]
        pred_val_bid = pred_val[bid, ...]
        cur_seg_bid = cur_seg_new[bid, ...]
        for oid in iou_oids:
            n_pred = np.sum((pred_val_bid == oid) * mask_bid)  # only the valid grids' pred
            # n_pred = np.sum(seg_pred_val == oid)
            n_gt = np.sum(cur_seg_bid == oid)
            n_intersect = np.sum(np.int32(cur_seg_bid == oid) * intersect_mask_bid)
            n_union = n_pred + n_gt - n_intersect
            iou_log += '_pred:' + str(n_pred) + '_gt:' + str(n_gt) + '_intersect:' + str(n_intersect) + '_union:' + str(n_union) + '_'
            if n_union == 0:
                total_iou += 1
                iou_log += '_:1\n'
            else:
                total_iou += n_intersect * 1.0 / n_union  # sum across parts
                iou_log += '_:'+str(n_intersect*1.0/n_union)+'\n'

        avg_iou = total_iou / len(iou_oids)  # average iou across parts, for one object
        # pdb.set_trace()
        acc_ops['total_mask_assign_acc_iou'] += avg_iou
    return avg_iou, acc_per_instance, iou_log


def assign_mask_to_vox(mask_pred, box, part_id, vox_size, part_num):
    """ apply mask to shape voxels
        input:  Unet_FCN_pred,
                mask_pred, box, part_id
                pts
        output [vsize,vsize,vsize,part_num], values from 0~1, float

        # # TODO: need combine box mask first, while the proposals of the same part are too much
    """
    # pdb.set_trace()
    pred_mask = np.transpose(np.squeeze(mask_pred), (0,4,1,2,3))
    box = np.squeeze(box)  # [num_rois, 6], normalized
    # [proposals, n_ch, h,w,d]
    part_id = np.squeeze(part_id)
    positive_ix = np.where(part_id > 0)
    # pdb.set_trace()
    # positive_part_ids = part_id[positive_ix].astype(np.int64) - 1
    positive_part_ids = part_id[positive_ix].astype(np.int64)
    box = np.squeeze(box[positive_ix, ...])
    pred_mask = np.squeeze(pred_mask[positive_ix, positive_part_ids, ...])
    # pred_mask: [num_proposals, h,w,d]
    # pred_mask = np.transpose(pred_mask, (1,2,3,0))
    # create voxel seg matrix
    vox_seg = np.zeros((vox_size,vox_size,vox_size,part_num+1))
    # fit the normalized box to target voxels shape
    box = np.round(box * (vox_size-1))
    # expand mini_masks to voxel size
    for n in range(pred_mask.shape[0]):
        cur_part = positive_part_ids[n]
        full_mask = my_util.expand_mask(box[n], pred_mask[n], vox_size)
        update_idx = full_mask >= vox_seg[:,:,:,cur_part]
        vox_seg[:,:,:,cur_part] = vox_seg[:,:,:,cur_part]*np.logical_not(update_idx)+full_mask*(full_mask>=0.35)*update_idx
        # vox_seg[:,:,:,cur_part] += full_mask * (full_mask>=0.5)
        # # TODO: need combine box mask first, while the proposals of the same part are too much
    return vox_seg  # [vsize,vsize,vsize,part_num], 0~1 float num


def assess_mask_result(target_mask, target_pid, pred_mask_val, acc_ops):
    """ evaluate mask pred miou """
    # acc_ops['loss_mask_sum'] += loss_mask_val
    # masks1, masks2: [Height, Width, Depth, instances]
    gt_mask_batch = target_mask
    gt_partid_batch = target_pid
    # pdb.set_trace()
    BATCH_SIZE = pred_mask_val.shape[0]
    for b in range(BATCH_SIZE):
        pred_mask = np.transpose(np.squeeze(pred_mask_val[b]), (0,4,1,2,3))
        # [proposals, n_ch, h,w,d]
        gt_mask = np.squeeze(gt_mask_batch[b])  # [num_rois, h,w,d]
        gt_partid = np.squeeze(gt_partid_batch[b])
        positive_ix = np.where(gt_partid > 0)
        positive_part_ids = gt_partid[positive_ix].astype(np.int64) - 1
        pred_mask = np.squeeze(pred_mask[positive_ix, positive_part_ids, ...])
        gt_mask = np.squeeze(gt_mask[positive_ix, ...])
        # pred_mask: [num_proposals, h,w,d]
        if len(pred_mask.shape) == 3:
            np.expand_dims(pred_mask, axis=-1)
            np.expand_dims(gt_mask, axis=-1)
        elif len(pred_mask.shape) == 4:
            pred_mask = np.transpose(pred_mask, (1,2,3,0))
            gt_mask = np.transpose(gt_mask, (1,2,3,0))
        else:
            print('Error!')
            pdb.set_trace()
        overlaps = my_util.compute_overlaps_masks(pred_mask, gt_mask)
        miou = np.trace(overlaps) / overlaps.shape[0]
        acc_ops['mask_miou'] += miou
    return miou, overlaps, np.squeeze(positive_ix), positive_part_ids


def assess_pts_result(pts, plb_pred, plb_gt, part_num, acc_ops):
    """ evaluate pts pred miou """
    # iou_oids = range(part_num)
    iou_oids = range(1, part_num+1, 1)
    pts, indices = np.unique(pts, return_index=True, axis=0)
    plb_gt = plb_gt[indices]
    plb_pred = plb_pred[indices]
    # print(pts.shape, plb_gt.shape, plb_pred.shape)

    mask = plb_gt == plb_pred
    # iou pts data
    total_iou_pts = 0.0
    iou_log_pts = 'iou_pts\n'  # iou details string

    for oid in iou_oids:
        n_pred = np.sum(plb_pred == oid)  # only the valid grids' pred
        # n_pred = np.sum(seg_pred_val == oid)
        n_gt = np.sum(plb_gt == oid)
        n_intersect = np.sum(np.int32(plb_pred == oid) * mask)
        n_union = n_pred + n_gt - n_intersect
        iou_log_pts += '_pred:' + str(n_pred) + '_gt:' + str(n_gt) + '_intersect:' + str(n_intersect) + '_union:' + str(n_union) + '_'
        if n_union == 0:
            total_iou_pts += 1
            iou_log_pts += '_:1\n'
        else:
            total_iou_pts += n_intersect * 1.0 / n_union
            iou_log_pts += '_:'+str(n_intersect*1.0/n_union)+'\n'

    avg_iou_pts = total_iou_pts / len(iou_oids)  # average iou across parts, for one object
    acc_ops['total_acc_iou_pts'] += avg_iou_pts
    acc_per_instance_pts = np.sum(mask) / len(plb_gt)
    acc_ops['total_accuracy_pts'] += acc_per_instance_pts
    return avg_iou_pts, acc_per_instance_pts, iou_log_pts
