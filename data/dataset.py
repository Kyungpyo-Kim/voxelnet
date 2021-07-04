import pathlib
import os
import glob
from torch.utils.data import Dataset
import numpy as np
import cv2
from utils import *
from box_overlaps import bbox_overlaps

class KittiDataset(Dataset):
    def __init__(self, data_dir, cfg, shuffle=False, aug=False,
                 is_testset=False) -> None:
        super().__init__()
        self.data_dir = data_dir

        # maxiumum number of points per voxel
        self.T = cfg.T
        # voxel size
        self.vd = cfg.vd
        self.vh = cfg.vh
        self.vw = cfg.vw
        # points cloud range
        self.xrange = cfg.xrange
        self.yrange = cfg.yrange
        self.zrange = cfg.zrange
        #   anchors: (200, 176, 2, 7) x y z h w l r
        self.anchors = cfg.anchors.reshape(-1, 7)
        self.anchors_per_position = cfg.anchors_per_position
        # voxel grid
        self.feature_map_shape = (int(cfg.H / 2), int(cfg.W / 2))
        # iou threshold
        self.pos_threshold = cfg.pos_threshold
        self.neg_threshold = cfg.neg_threshold

        self.shuffle = shuffle
        self.aug = aug
        self.is_testset = is_testset

        """
        Get pathes of dataset
        """
        self.f_rgb = glob.glob(os.path.join(self.data_dir, 'image_2', '*.png'))
        self.f_lidar = glob.glob(os.path.join(
            self.data_dir, 'velodyne', '*.bin'))
        self.f_label = glob.glob(os.path.join(
            self.data_dir, 'label_2', '*.txt'))
        self.f_calib = glob.glob(os.path.join(
            self.data_dir, 'calib', '*.txt'))

        self.f_rgb.sort()
        self.f_lidar.sort()
        self.f_label.sort()
        self.f_calib.sort()

        self.data_tag = [name.split('/')[-1].split('.')[-2]
                         for name in self.f_rgb]

        assert len(self.data_tag) != 0, 'Dataset folder is not correct!'
        assert len(self.data_tag) == len(self.f_rgb) == len(
            self.f_lidar) == len(self.f_calib), 'Dataset folder is not correct!'

        nums = len(self.f_rgb)
        self.indices = list(range(nums))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        # [voxel_features, voxel_coords, pos_equal_one, neg_equal_one, targets, images, calibs, ids]
        pointcloud = self.pointcloudFromFile(self.f_lidar[index])
        voxel_features, voxel_coords = self.voxelize(pointcloud)
        calib = self.calibFromFile(self.f_calib[index])
        
        data = {}
        data["voxel_features"] = voxel_features
        data["voxel_coords"] = voxel_coords
        data["rgb"] = cv2.imread(self.f_rgb[index])
        data["pointcloud"] = pointcloud
        data["calib"] = calib
        data["file_id"] = pathlib.Path(self.f_calib[index]).stem

        if not self.is_testset:
            label = self.labelFromFile(self.f_label[index], calib['Tr_velo2cam'])
            pos_equal_one, neg_equal_one, targets = self.calculateTarget(label)
            data["label"] = label
            data["pos_equal_one"] = pos_equal_one
            data["neg_equal_one"] = neg_equal_one
            data["target"] = targets
        else:
            data["label"] = None
            data["pos_equal_one"] = None
            data["neg_equal_one"] = None
            data["target"] = None

        return data

    def __len__(self):
        return len(self.indices)

    def voxelize(self, pointcloud):
        """
        TODO Move to utils package
        """
        # shuffling the points
        np.random.shuffle(pointcloud)

        # order: B, W, H, D
        voxel_coords = ((pointcloud[:, :3] - np.array([self.xrange[0], self.yrange[0], self.zrange[0]])) / (
                        self.vw, self.vh, self.vd)).astype(np.int32)
        # convert to  (B, D, H, W)
        voxel_coords = voxel_coords[:, [2, 1, 0]]

        voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0,
                                                        return_inverse=True, return_counts=True)
        voxel_features = []

        for i in range(len(voxel_coords)):
            voxel = np.zeros((self.T, 7), dtype=np.float32)
            pts = pointcloud[inv_ind == i]
            if voxel_counts[i] > self.T:
                pts = pts[:self.T, :]
                voxel_counts[i] = self.T

            # augment the points with nomalized points
            pts_nomalized = pts[:, :3] - np.mean(pts[:, :3], 0)
            voxel[:pts.shape[0], :] = np.concatenate(
                (pts, pts_nomalized), axis=1)
            voxel_features.append(voxel)

        return np.array(voxel_features), voxel_coords

    def calibFromFile(self, file):
        """
        load projection matrix
        """
        with open(file) as fi:
            lines = fi.readlines()
            assert (len(lines) == 8)

        obj = lines[0].strip().split(' ')[1:]
        P0 = np.array(obj, dtype=np.float32)
        obj = lines[1].strip().split(' ')[1:]
        P1 = np.array(obj, dtype=np.float32)
        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)
        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)
        obj = lines[6].strip().split(' ')[1:]
        Tr_imu_to_velo = np.array(obj, dtype=np.float32)
        return {'P2': P2.reshape(3, 4),
                'R0': R0.reshape(3, 3),
                'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4),
                'Tr_imu_to_velo': Tr_imu_to_velo.reshape(3, 4)}

    def pointcloudFromFile(self, file):
        return np.fromfile(file, dtype=np.float32).reshape(-1, 4)

    def labelFromFile(self, file, Tr):
        with open(file, 'r') as f:
            lines = f.readlines()

        gt_boxes3d_corner = []

        num_obj = len(lines)

        def box3d_cam_to_velo(box3d, Tr):

            def project_cam2velo(cam, Tr):
                T = np.zeros([4, 4], dtype=np.float32)
                T[:3, :] = Tr
                T[3, 3] = 1
                T_inv = np.linalg.inv(T)
                lidar_loc_ = np.dot(T_inv, cam)
                lidar_loc = lidar_loc_[:3]
                return lidar_loc.reshape(1, 3)

            def ry_to_rz(ry):
                angle = -ry - np.pi / 2

                if angle >= np.pi:
                    angle -= np.pi
                if angle < -np.pi:
                    angle = 2*np.pi + angle

                return angle

            h, w, l, tx, ty, tz, ry = [float(i) for i in box3d]
            cam = np.ones([4, 1])
            cam[0] = tx
            cam[1] = ty
            cam[2] = tz
            t_lidar = project_cam2velo(cam, Tr)

            Box = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                            [w / 2, -w / 2, -w / 2, w / 2,
                                w / 2, -w / 2, -w / 2, w / 2],
                            [0, 0, 0, 0, h, h, h, h]])

            rz = ry_to_rz(ry)

            rotMat = np.array([
                [np.cos(rz), -np.sin(rz), 0.0],
                [np.sin(rz), np.cos(rz), 0.0],
                [0.0, 0.0, 1.0]])

            velo_box = np.dot(rotMat, Box)

            cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T

            box3d_corner = cornerPosInVelo.transpose()

            return box3d_corner.astype(np.float32)

        for j in range(num_obj):
            obj = lines[j].strip().split(' ')

            obj_class = obj[0].strip()
            if obj_class not in cfg.class_list:
                continue

            box3d_corner = box3d_cam_to_velo(obj[8:], Tr)

            gt_boxes3d_corner.append(box3d_corner)

        gt_boxes3d_corner = np.array(gt_boxes3d_corner).reshape(-1, 8, 3)

        return gt_boxes3d_corner

    def calculateTarget(self, label):
        # Input:
        #   labels: (N,)
        #   feature_map_shape: (w, l)
        #   anchors: (w, l, 2, 7)
        # Output:
        #   pos_equal_one (w, l, 2)
        #   neg_equal_one (w, l, 2)
        #   targets (w, l, 14)
        # attention: cal IoU on birdview
        anchors_d = np.sqrt(self.anchors[:, 4] ** 2 + self.anchors[:, 5] ** 2)
        pos_equal_one = np.zeros((*self.feature_map_shape, 2))
        neg_equal_one = np.zeros((*self.feature_map_shape, 2))
        targets = np.zeros((*self.feature_map_shape, 14))

        gt_xyzhwlr = box3d_corner_to_center_batch(label)
        anchors_corner = anchors_center_to_corner(self.anchors)
        anchors_standup_2d = corner_to_standup_box2d_batch(anchors_corner)

        # BOTTLENECK
        gt_standup_2d = corner_to_standup_box2d_batch(label)

        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )

        id_highest = np.argmax(iou.T, axis=1)  # the maximum anchor's ID
        id_highest_gt = np.arange(iou.T.shape[0])
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]
        # find anchor iou > cfg.XXX_POS_IOU
        id_pos, id_pos_gt = np.where(iou > self.pos_threshold)
        # find anchor iou < cfg.XXX_NEG_IOU
        id_neg = np.where(np.sum(iou < self.neg_threshold,
                                 axis=1) == iou.shape[1])[0]

        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
        # TODO: uniquify the array in a more scientific way
        id_pos, index = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()
        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(
            id_pos, (*self.feature_map_shape, self.anchors_per_position))
        pos_equal_one[index_x, index_y, index_z] = 1
        # ATTENTION: index_z should be np.array

        targets[index_x, index_y, np.array(index_z) * 7] = \
            (gt_xyzhwlr[id_pos_gt, 0] - self.anchors[id_pos, 0]) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = \
            (gt_xyzhwlr[id_pos_gt, 1] - self.anchors[id_pos, 1]) / anchors_d[id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = \
            (gt_xyzhwlr[id_pos_gt, 2] - self.anchors[id_pos, 2]) / self.anchors[id_pos, 3]
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
            gt_xyzhwlr[id_pos_gt, 3] / self.anchors[id_pos, 3])
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
            gt_xyzhwlr[id_pos_gt, 4] / self.anchors[id_pos, 4])
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
            gt_xyzhwlr[id_pos_gt, 5] / self.anchors[id_pos, 5])
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
                gt_xyzhwlr[id_pos_gt, 6] - self.anchors[id_pos, 6])
        index_x, index_y, index_z = np.unravel_index(
            id_neg, (*self.feature_map_shape, self.anchors_per_position))
        neg_equal_one[index_x, index_y, index_z] = 1
        # to avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(
            id_highest, (*self.feature_map_shape, self.anchors_per_position))
        neg_equal_one[index_x, index_y, index_z] = 0

        return pos_equal_one, neg_equal_one, targets



if __name__ == "__main__":
    """
    for import config, run with module options
    $python3 -m data.dataset
    """
    from config import config as cfg

    dataset_training = KittiDataset(pathlib.Path(
        __file__).parent.absolute()/"training", cfg)

    print("training dataset length: ", len(dataset_training))

    idx = np.random.choice(len(dataset_training), 1).item()

    print("training dataset sample {}: ".format(idx), dataset_training[idx])
