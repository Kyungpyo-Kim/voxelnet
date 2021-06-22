import pathlib
import os
import glob
from torch.utils.data import Dataset
import numpy as np
import cv2


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
        print(self.f_rgb[index])
        print(self.f_lidar[index])
        print(self.f_label[index])
        print(self.f_calib[index])

        pointcloud = self.pointcloudFromFile(self.f_lidar[index])
        voxel_features, voxel_coords = self.voxelize(pointcloud)
        calib = self.calibFromFile(self.f_calib[index])
        label = self.labelFromFile(self.f_label[index], calib['Tr_velo2cam'])

        data = {}
        data["voxel_features"] = voxel_features
        data["voxel_coords"] = voxel_coords
        data["pos_equal_one"] = None
        data["neg_equal_one"] = None
        data["target"] = None
        data["rgb"] = cv2.imread(self.f_rgb[index])
        data["pointcloud"] = pointcloud
        data["calib"] = calib
        data["id"] = None
        data["label"] = label

        if not self.is_testset:
            data["label"] = None
        else:
            data["label"] = None

        return data

    def __len__(self):
        return len(self.indices)

    def voxelize(self, pointcloud):
        # shuffling the points
        np.random.shuffle(pointcloud)

        # order: W, H, D
        voxel_coords = ((pointcloud[:, :3] - np.array([self.xrange[0], self.yrange[0], self.zrange[0]])) / (
                        self.vw, self.vh, self.vd)).astype(np.int32)
        # convert to  (D, H, W)
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
