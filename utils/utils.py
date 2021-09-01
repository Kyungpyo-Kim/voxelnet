import numpy as np
import numba


def getBBoxCorners3D(x, y, z, w, l, h, yaw):
    """
    w, h, l, x, y, z, yaw => 8 corners(3,8)
    """
    # compute rotational matrix around yaw axis
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                  [np.sin(yaw),  np.cos(yaw), 0],
                  [0,            0,           1]])

    # 3D bounding box corners
    x_corners = np.array([0, 0, w, w, 0, 0, w, w]) - w/2
    y_corners = np.array([0, l, l, 0, 0, l, l, 0]) - l/2
    z_corners = np.array([0, 0, 0, 0, h, h, h, h]) - h/2

    # bounding box in object co-ordinate
    corners = np.array([x_corners, y_corners, z_corners])

    # rotate
    corners = R.dot(corners)

    # translate
    corners += np.array([x, y, z]).reshape((3, 1))

    return corners


def getBBoxCorners3DFromDict(data):
    x = data['x']
    y = data['y']
    z = data['z']
    w = data['w']
    l = data['l']
    h = data['h']
    yaw = data['yaw']
    return getBBoxCorners3D(x, y, z, w, l, h, yaw)


def getBBoxCorners3DFromList(data):
    """
    arg: data (7,)
    """
    x, y, z, w, l, h, yaw = data
    return getBBoxCorners3D(x, y, z, w, l, h, yaw)


def getBBoxCorners3DFromBatch(data):
    """"
    arg: data (B, 7)
    return corners (B, 3, 8)
    """
    corners = np.zeros((data.shape[0], 3, 8))
    for i in range(data.shape[0]):
        corners[i] = getBBoxCorners3DFromList(data[i])
    return corners


def getBBoxStandupCorners2DBatch(data):
    """
    arg: data (B, 7)
    return corners (B, 4) with [x min, y min, x max, y max]
    """
    B = data.shape[0]
    corners = np.zeros((B, 4))
    corners3d = getBBoxCorners3DFromBatch(data)
    corners[:, 0] = np.min(corners3d[:, 0, :], axis=1)
    corners[:, 1] = np.min(corners3d[:, 1, :], axis=1)
    corners[:, 2] = np.max(corners3d[:, 0, :], axis=1)
    corners[:, 3] = np.max(corners3d[:, 1, :], axis=1)
    return corners


def voxelize(pointcloud, range_x, range_y, range_z,
             voxel_size_x, voxel_size_y, voxel_size_z, voxel_pt_num,
             shuffle=True):
    """
    param: pointcloud (B 4)
    return: 
        voxel_feature (B T 7), where T is number of points in a voxel 
        voxel_coords (B Vx Vy Vz)

    """
    # shuffling the points
    if shuffle:
        np.random.shuffle(pointcloud)

    # point cloud order: B, X, Y, Z, I
    # voxel_coords order: B, Vx, Vy, Vz
    voxel_coords = ((pointcloud[:, :3] - np.array([range_x[0], range_y[0], range_z[0]])) / (
        voxel_size_x, voxel_size_y, voxel_size_z)).astype(np.int32)
    
    voxel_coords, inv_ind, voxel_counts = np.unique(
        voxel_coords, axis=0, return_inverse=True, return_counts=True)

    # voxel feature: x, y, z, i, x_norm, y_norm, z_norm
    voxel_features = []
    for i in range(len(voxel_coords)):
        voxel = np.zeros((voxel_pt_num, 7), dtype=np.float32)
        pts = pointcloud[inv_ind == i]
        if voxel_counts[i] > voxel_pt_num:
            pts = pts[:voxel_pt_num, :]
            voxel_counts[i] = voxel_pt_num

        # augment the points with nomalized points
        pts_nomalized = pts[:, :3] - np.mean(pts[:, :3], 0)
        voxel[:pts.shape[0], :] = np.concatenate(
            (pts, pts_nomalized), axis=1)
        voxel_features.append(voxel)

    return np.array(voxel_features), voxel_coords


@numba.jit(nopython=True)
def calculateIouJit(boxes, query_boxes, eps=1.0):
    """
    reference: https://github.com/traveller59/second.pytorch/
    calculate box iou. note that jit version runs 2x faster than cython in 
    my machine!
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + eps) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + eps))
        for n in range(N):
            # x, W, w
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]) + eps)
            if iw > 0:
                # y, H, l
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(
                    boxes[n, 1], query_boxes[k, 1]) + eps)
                if ih > 0:
                    ua = (
                        (boxes[n, 2] - boxes[n, 0] + eps) *
                        (boxes[n, 3] - boxes[n, 1] + eps) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua
    return overlaps


@numba.jit(nopython=True)
def anchorEncoder(anchor, target):
    """
    arg: anchor (N, 7)
    arg: target (N, 7)
    return target (N, 7)
    """
    anchors_d = np.sqrt(
        anchor[:, 0] ** 2 + anchor[:, 0] ** 2)

    target[:, 0] = (target[:, 0] - anchor[:, 0]) / anchors_d
    target[:, 1] = (target[:, 1] - anchor[:, 1]) / anchors_d
    target[:, 2] = (target[:, 2] - anchor[:, 2]) / anchor[:, 5]
    target[:, 3] = np.log(target[:, 3] / anchor[:, 3])
    target[:, 4] = np.log(target[:, 4] / anchor[:, 4])
    target[:, 5] = np.log(target[:, 5] / anchor[:, 5])
    target[:, 6] = target[:, 6] - anchor[:, 6]
    return target


@numba.jit(nopython=True)
def anchorDecoder(anchor, target):
    """
    arg: anchor (N, 7)
    arg: target (N, 7)
    return target (N, 7)
    """
    anchors_d = np.sqrt(
        anchor[:, 0] ** 2 + anchor[:, 0] ** 2)

    target[:, 0] = target[:, 0] * anchors_d + anchor[:, 0]
    target[:, 1] = target[:, 1] * anchors_d + anchor[:, 1]
    target[:, 2] = target[:, 2] * anchor[:, 5] + anchor[:, 2]
    target[:, 3] = np.exp(target[:, 3]) * anchor[:, 3]
    target[:, 4] = np.exp(target[:, 4]) * anchor[:, 4]
    target[:, 5] = np.exp(target[:, 5]) * anchor[:, 5]
    target[:, 6] = target[:, 6] + anchor[:, 6]
    return target
