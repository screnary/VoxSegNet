""" Utility functions for processing point clouds.

Author: Charles R. Qi, Hao Su
Date: November 2016
"""
import pdb
import os
import sys
from scipy import ndimage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Draw point cloud
from eulerangles import euler2mat

# Point cloud IO
import numpy as np
from plyfile import PlyData, PlyElement


# ----------------------------------------
# Point Cloud/Volume Conversions
# ----------------------------------------
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def point_cloud_to_volume_batch(point_clouds, vsize=12, radius=1.0, flatten=False):
    """ Input is BxNx3 batch of point cloud
        Output is Bx(vsize^3)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume(np.squeeze(point_clouds[b,:,:]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)


def point_cloud_to_volume(points, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
        (actually, after nomalization, all points are in a unit sphere)
    """
    vol = np.zeros((vsize,vsize,vsize))
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel  # shift all points to non-negative coordinates, assign to an occupancy grid
    locations = locations.astype(int)
    vol[locations[:,0],locations[:,1],locations[:,2]] = 1.0
    return vol

# a = np.zeros((16,1024,3))
# print point_cloud_to_volume_batch(a, 12, 1.0, False).shape


def point_cloud_to_volume_batch_wzj_v2(point_clouds, part_labels, ptsnum, vsize=12, radius=1.0, flatten=False):  # for voxel data generation--[wzj 201704]
    """ 20180402--for points data stacked [Totalnum, 3]
    So, should process to:
        Input is BxNx3 batch of point cloud, and its corresponding BxN part labels
        Output is Bx(vsize^3) vol_list and its pid_vol_list
    """
    end = np.cumsum(ptsnum)
    start = np.concatenate(([0], end[:-1]), axis=0)

    vol_list = []
    pid_vol_list = []
    for b in range(ptsnum.shape[0]):
        if (b + 1) % 10 == 0:
            print(b+1, '/', ptsnum.shape[0])
        st = start[b]
        ed = end[b]
        [vol, pid_vol] = point_cloud_to_volume_wzj(np.squeeze(point_clouds[st:ed,:]), np.squeeze(part_labels[st:ed]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
            pid_vol_list.append(pid_vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
            pid_vol_list.append(np.expand_dims(np.expand_dims(pid_vol, -1), 0))
    if flatten:
        return np.vstack(vol_list), np.vstack(pid_vol_list)
    else:
        return np.concatenate(vol_list, 0), np.concatenate(pid_vol_list, 0)


def point_cloud_to_volume_batch_wzj(point_clouds, part_labels, vsize=12, radius=1.0, flatten=False):  # for voxel data generation--[wzj 201704]
    """ Input is BxNx3 batch of point cloud, and its corresponding BxN part labels
        Output is Bx(vsize^3) vol_list and its pid_vol_list
    """
    vol_list = []
    pid_vol_list = []
    for b in range(point_clouds.shape[0]):
        if (b + 1) % 100 == 0:
            print(b+1, '/', point_clouds.shape[0])
        [vol, pid_vol] = point_cloud_to_volume_wzj(np.squeeze(point_clouds[b,:,:]), np.squeeze(part_labels[b,:]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
            pid_vol_list.append(pid_vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
            pid_vol_list.append(np.expand_dims(np.expand_dims(pid_vol, -1), 0))
    if flatten:
        return np.vstack(vol_list), np.vstack(pid_vol_list)
    else:
        return np.concatenate(vol_list, 0), np.concatenate(pid_vol_list, 0)


def point_cloud_to_volume_wzj(points, part_label, vsize, radius=1.0):
    """ input is Nx3 points, and corresponding N (Nx1 array) part labels (per point, start from 0).
        output is vsize*vsize*vsize occupancy grid---vol;
        and its vsize*vsize*vsize part label value---pid_vol; pid_vol: 0 for background, others is 'pid + 1'
    """
    normalize_Flag = True
    if normalize_Flag:
        points = pc_normalize(points)
    vol = np.zeros((vsize,vsize,vsize))
    pid_vol = np.zeros((vsize,vsize,vsize))
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel  # shift all points to non-negative coordinates, assign to an occupancy grid
    locations = locations.astype(int)
    locations[locations > (vsize - 1)] = vsize - 1
    locations[locations < 0] = 0
    vol[locations[:,0],locations[:,1],locations[:,2]] = 1.0
    loca_unique = np.unique(locations.view(np.dtype((np.void, locations.dtype.itemsize*locations.shape[1])))).view(locations.dtype).reshape(-1, locations.shape[1])
    # pdb.set_trace()
    for loca in loca_unique:
        idx = np.sum(np.abs(locations - loca), -1) == 0
        pids = part_label[idx]
        [pids_unique, counts] = np.unique(pids, return_counts=True)
        part_id = pids_unique[np.argmax(counts)]
        pid_vol[loca[0], loca[1], loca[2]] = part_id + 1  # background is 0
    mask_pid = pid_vol > 0
    mask_vol = vol > 0
    if np.sum(mask_pid != mask_vol) != 0:
        print("Error! vol_pid and vol not match\n")
    return vol, pid_vol


def down_volume_seg_batch_wzj(segs, radius=1.0, flatten=False):  # for voxel data generation--[wzj 201704]
    """ Input is BxNx3 batch of point cloud, and its corresponding BxN part labels
        Output is Bx(vsize^3) vol_list and its pid_vol_list
    """
    pid_vol_list = []
    for b in range(segs.shape[0]):
        pid_vol = down_volume_seg_wzj(np.squeeze(segs[b,...]), radius)
        if flatten:
            pid_vol_list.append(pid_vol.flatten())
        else:
            pid_vol_list.append(np.expand_dims(pid_vol, 0))
    if flatten:
        return np.vstack(pid_vol_list)
    else:
        return np.concatenate(pid_vol_list, 0)


def down_volume_seg_wzj(seg, radius=1.0):
    """ input is Nx3 points, and corresponding N (Nx1 array) part labels (per point).
        output is vsize*vsize*vsize occupancy grid---vol;
        and its vsize*vsize*vsize part label value---pid_vol; pid_vol: 0 for background, others is 'pid + 1'
    """
    seg = np.squeeze(seg)
    vol_in = seg > 0  # 0 is background
    vsize = int(seg.shape[0] / 2)
    points, part_label = volume_seg_to_list(vol_in, seg)
    points = pc_normalize(points)
    vol = np.zeros((vsize,vsize,vsize))
    pid_vol = np.zeros((vsize,vsize,vsize))  # background is zero
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel  # shift all points to non-negative coordinates, assign to an occupancy grid
    locations = locations.astype(int)
    vol[locations[:,0],locations[:,1],locations[:,2]] = 1.0
    loca_unique = np.unique(locations.view(np.dtype((np.void, locations.dtype.itemsize*locations.shape[1])))).view(locations.dtype).reshape(-1, locations.shape[1])
    # pdb.set_trace()
    for loca in loca_unique:
        idx = np.sum(np.abs(locations - loca), -1) == 0
        pids = part_label[idx]
        [pids_unique, counts] = np.unique(pids, return_counts=True)
        part_id = pids_unique[np.argmax(counts)]
        pid_vol[loca[0], loca[1], loca[2]] = part_id  # same part label value as the input large seg matrix
    mask_pid = pid_vol > 0
    mask_vol = vol > 0
    if np.sum(mask_pid != mask_vol) != 0:
        print("Error! vol_pid and vol not match\n")
    return pid_vol


""" for 3DCNN RoI Layer [wzj-20171009] """


def gen_loc_centers_batch(volumes, is_training, loc_train=500, loc_test=4000):
    """ input: volume (B,V,V,V[,1])
        output: loc=[num_loc, 3], label=[num_loc]
    """
    if (is_training):
        num_loc = loc_train
    else:
        num_loc = loc_test

    locs = []
    # labels = []
    volumes = np.squeeze(volumes)  # shape is (B,V,V,V)
    # seg_labels = np.squeeze(seg_labels)  # shape is (B,V,V,V)
    for b in range(volumes.shape[0]):
        volume = np.squeeze(volumes[b,...])
        # seg = np.squeeze(seg_labels[b,...])
        loc = gen_loc_centers(volume, num_loc)  # loc shape (num_loc,3); label shape (num_loc,)
        locs.append(np.expand_dims(loc, axis=0))
        # labels.append(np.expand_dims(label, axis=0))

    return np.concatenate(locs, 0)  # , np.concatenate(labels, 0)  # shape (B, num_loc, 3), (B, num_loc)


def gen_loc_centers(volume, num_loc):
    volume = np.squeeze(volume)  # shape is (V,V,V)
    loc_occ = np.transpose(np.nonzero(volume))
    num_occ = np.count_nonzero(volume)
    if num_occ < num_loc:
        loc_occ = np.concatenate((loc_occ, loc_occ))
        loc_occ = loc_occ[:num_loc, ...]
        idx = np.arange(num_loc)
    else:
        idx = np.arange(num_occ)

    np.random.shuffle(idx)
    loc_res = loc_occ[idx, ...]
    loc = loc_res[:num_loc, ...]  # shape (num_loc, 3)
    # label = seg_label[loc[:,0], loc[:,1], loc[:,2]]  # shape (num_loc,)

    return loc  # , label


""" End of [20171009] """

""" Modified [20171015] for saving computation memory"""


def gen_loc_centers_batch2(volumes):
    """ input: volume (B,V,V,V[,1])
        output: loc=[num_loc, 3], label=[num_loc]
    """
    locs = []
    loc_count = []
    # labels = []
    # volumes = np.squeeze(volumes)  # shape is (B,V,V,V)
    # seg_labels = np.squeeze(seg_labels)  # shape is (B,V,V,V)
    for b in range(volumes.shape[0]):
        volume = np.squeeze(volumes[b,...])
        # seg = np.squeeze(seg_labels[b,...])
        loc, num_loc = gen_loc_centers2(volume)  # loc shape (num_loc,3); label shape (num_loc,)
        # locs.append(np.expand_dims(loc, axis=0))
        locs.append(loc)
        loc_count.append(num_loc)
        # labels.append(np.expand_dims(label, axis=0))

    return locs, loc_count  # , np.concatenate(labels, 0)  # shape (B, num_loc, 3), (B, num_loc)


def gen_loc_centers2(volume):
    volume = np.squeeze(volume)  # shape is (V,V,V)
    loc_occ = np.transpose(np.nonzero(volume))
    num_occ = np.count_nonzero(volume)
    idx = np.arange(num_occ)
    np.random.shuffle(idx)
    loc_res = loc_occ[idx, ...]
    # label = seg_label[loc[:,0], loc[:,1], loc[:,2]]  # shape (num_loc,)

    return loc_res, num_occ  # , label


def get_slice(volume, loc, w_s):
    """ Inputs:
        volume is [v_size,v_size,v_size,n_ch]
        loc is [x,y,z]: occupied volume grid coordinate
        w_s is window size: 3/5/7/..., np.int
        Output:
        slice_res is [w_s,w_s,w_s,n_ch]
    """
    volume = np.squeeze(volume)
    v_s = volume.shape[0]
    n_ch = volume.shape[-1]
    slice_res = np.zeros((w_s, w_s, w_s, n_ch), dtype=np.float32)

    shift = np.int((w_s - 1) / 2)
    start = loc - shift
    end = loc + shift + 1
    st_valid = start + 0
    ed_valid = end + 0
    store_st = np.zeros((3,), dtype=np.int)
    store_ed = np.ones((3,), dtype=np.int) * w_s
    if (start < 0).sum():
        st_valid[start < 0] = 0
        store_st = np.abs(start * (start < 0))
    if (end > v_s).sum():
        ed_valid[end > v_s] = v_s
        store_ed = store_ed - (end - v_s) * (end > v_s)
    slice_res[store_st[0]:store_ed[0],
              store_st[1]:store_ed[1],
              store_st[2]:store_ed[2], ...] = volume[st_valid[0]:ed_valid[0], st_valid[1]:ed_valid[1], st_valid[2]:ed_valid[2], ...]
    return slice_res

# # test
# volume = np.random.randn(5,5,5,3)

# S = get_slice(volume, np.array([4,4,0]), np.int(3))
# pdb.set_trace()


""" End of [20171015] """


def part_boundary_detection(vol_seg):
    """ input : vol_seg is occupancy grid with part label value
        return: vol_boundary. --wzj 201704
            0 vacancy; 1 boundary; 2 non boundary voxel
    """
    vol_seg = np.squeeze(vol_seg)  # [vol_size,vol_size,vol_size]
    vol_bon = np.zeros(vol_seg.shape)  # vol_boundary
    vsize = vol_seg.shape[0]
    win = 3  # sliding window size
    offset = (win - 1) / 2  # offset==1
    x,y,z = np.where(vol_seg != 0)  # valid voxels
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    z = z.astype(np.int32)
    vol_bon[x,y,z] = 2  # non boundary voxel
    x_l = (x - offset).astype(np.int32)  # lower location
    x_h = (x + offset).astype(np.int32)  # higher location
    y_l = (y - offset).astype(np.int32)
    y_h = (y + offset).astype(np.int32)
    z_l = (z - offset).astype(np.int32)
    z_h = (z + offset).astype(np.int32)

    x_l[x_l < 0] = 0  # trancate x
    x_h[x_h > vsize] = vsize
    y_l[y_l < 0] = 0  # trancate y
    y_h[y_h > vsize] = vsize
    z_l[z_l < 0] = 0  # trancate z
    z_h[z_h > vsize] = vsize

    print('occupied grids: ', len(x))
    for a in range(len(x)):
        # pdb.set_trace()
        slide_box = vol_seg[x_l[a]:x_h[a], y_l[a]:y_h[a], z_l[a]:z_h[a]]
        label_tab = np.unique(slide_box)
        tab = label_tab
        if(label_tab[0] == 0):  # if there are vacancy grids
            tab = label_tab[1:]
        if(len(tab) > 1):
            vol_bon[x[a], y[a], z[a]] = 1
    return vol_bon


def part_boundary_detection_batch(vol_segs, flatten=False):  # for voxel data generation--[wzj 201704]
    """ Input is BxNx3 batch of seg volumes,
        Output is Bx(vsize^3) vol_list boundary detection results
    """
    vol_bon_list = []
    print(vol_segs.shape)
    for b in range(vol_segs.shape[0]):  # batch
        print(b, '/', vol_segs.shape[0]-1)
        vol_bon = part_boundary_detection(np.squeeze(vol_segs[b,...]))
        if flatten:
            vol_bon_list.append(vol_bon.flatten())
        else:
            vol_bon_list.append(np.expand_dims(np.expand_dims(vol_bon, -1), 0))
    if flatten:
        return np.vstack(vol_bon_list)
    else:
        return np.concatenate(vol_bon_list, 0)


def volume_distance_field_batch(vols):
    """ Input is B*Vsize*Vsize*Vsize volumes,
        Output is B*(Vsize*Vsize*Vsize) vol_edt_list distance transform results
    """
    vol_edt_list = []
    for b in range(vols.shape[0]):
        vol_edt = volume_distance_field(np.squeeze(vols[b,...]))
        vol_edt_list.append(np.expand_dims(np.expand_dims(vol_edt, -1), 0))
    return np.concatenate(vol_edt_list, 0)


def volume_distance_field(vol):
    """ Input is Vsize*Vsize*Vsize voxel, (uint8)
        Output is Vsize*Vsize*Vsize distance field (float64)
    """
    vol = np.squeeze(vol)
    edt_input = vol == 0  # 0 as the obstacle voxel
    edt_out = ndimage.distance_transform_edt(edt_input)
    edt = np.exp(edt_out*-1)
    edt = edt/np.sqrt(2*np.var(edt))
    return edt


def volume_to_point_cloud(vol):
    """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
        return Nx3 numpy array.
    """
    vsize = vol.shape[0]
    assert(vol.shape[1] == vsize and vol.shape[2] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a,b,c] == 1:
                    points.append(np.array([a,b,c]))
    if len(points) == 0:
        return np.zeros((0,3))
    points = np.vstack(points)
    return points


def volume_seg_to_list(vol, seg):
    """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
        return Nx3 numpy array.
    """
    vsize = vol.shape[0]
    assert(vol.shape[1] == vsize and vol.shape[2] == vsize and vol.shape == seg.shape)
    points = []
    seg_label = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a,b,c] == 1:
                    points.append(np.array([a,b,c]))
                    seg_label.append(seg[a,b,c])
    if len(points) == 0:
        return np.zeros((0,3)), -1
    points = np.vstack(points)
    seg_label = np.vstack(seg_label)
    return points, seg_label

# ----------------------------------------
# Point cloud IO
# ----------------------------------------


def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array


def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


# ----------------------------------------
# Simple Point cloud and Volume Renderers
# ----------------------------------------

def draw_point_cloud(input_points, canvasSize=500, space=200, diameter=25,
                     xrot=0, yrot=0, zrot=0, switch_xyz=[0,1,2], normalize=False):  # normalize=True
    """ Render point cloud to image with alpha channel.
        Input:
            points: Nx3 numpy array (+y is up direction)
        Output:
            gray image as numpy array of size canvasSizexcanvasSize
    """
    image = np.zeros((canvasSize, canvasSize))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]
    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # Normalize the point cloud
    # We normalize scale to fit points in a unit sphere
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
        points /= furthest_distance

    # Pre-compute the Gaussian disk
    radius = (diameter-1)/2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i-radius) + (j-radius) * (j-radius) <= radius * radius:
                disk[i, j] = np.exp((-(i-radius)**2 - (j-radius)**2)/(radius**2))
    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]
    # dvv = disk > 0
    # dvv = dvv.astype(np.float)
    # pdb.set_trace()

    # Order points by z-buffer
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    # pdb.set_trace()
    if(np.max(points[:,2]) - np.min(points[:,2]) == 0):
        points[:, 2] = points[:, 2]
    else:
        points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
    
    max_depth = np.max(points[:, 2])

    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSize/2 + (x*space)
        yc = canvasSize/2 + (y*space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))

        px = dx + xc
        py = dy + yc
        # pdb.set_trace()
        image[px, py] = image[px, py] * 0.0 + dv * (max_depth-points[j, 2] + 0.9) * 1.0
        # image[px, py] = image[px, py] * 0.7 + dv * (max_depth - points[j, 2]) * 0.3

    # image = image / np.max(image)
    return image


def point_cloud_three_views(points):
    """ input points Nx3 numpy array (+y is up direction).
        return an numpy array gray image of size 500x1500. """
    # +y is up direction
    # xrot is azimuth
    # yrot is in-plane
    # zrot is elevation
    img1 = draw_point_cloud(points, zrot=110/180.0*np.pi, xrot=45/180.0*np.pi, yrot=0/180.0*np.pi)
    img2 = draw_point_cloud(points, zrot=70/180.0*np.pi, xrot=135/180.0*np.pi, yrot=0/180.0*np.pi)
    img3 = draw_point_cloud(points, zrot=180.0/180.0*np.pi, xrot=90/180.0*np.pi, yrot=0/180.0*np.pi)
    image_large = np.concatenate([img1, img2, img3], 1)
    return image_large


from PIL import Image
def point_cloud_three_views_demo():
    """ Demo for draw_point_cloud function """
    points = read_ply('../third_party/mesh_sampling/piano.ply')
    im_array = point_cloud_three_views(points)
    img = Image.fromarray(np.uint8(im_array*255.0))
    img.save('piano.jpg')


if __name__ == "__main__":
    point_cloud_three_views_demo()


import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def pyplot_draw_point_cloud(points, output_filename):
    """ points is a Nx3 numpy array """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig(output_filename)


def pyplot_draw_volume(vol, output_filename):
    """ vol is of size vsize*vsize*vsize
        output an image to output_filename
    """
    points = volume_to_point_cloud(vol)
    pyplot_draw_point_cloud(points, output_filename)
