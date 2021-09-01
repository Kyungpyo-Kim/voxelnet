import pathlib
from torch.utils.data import Dataset
import numpy as np
import data.kitti.utils as kitti_utils
import utils.utils as utils


def detection_collate(batch):
    voxel_features = []
    voxel_coords = []
    pos_equal_one = []
    neg_equal_one = []
    targets = []
    images = []
    pointclouds = []
    calibs = []
    ids = []

    for i, sample in enumerate(batch):
        voxel_features.append(sample['voxel_features'])
        voxel_coords.append(
            np.pad(sample['voxel_coords'], ((0, 0), (1, 0)), mode='constant',
                   constant_values=i
                   )
        )
        pos_equal_one.append(sample['pos_equal_one'])
        neg_equal_one.append(sample['neg_equal_one'])
        targets.append(sample['target'])
        images.append(sample['rgb'])
        pointclouds.append(sample['pointcloud'])
        calibs.append(sample['calib'])
        ids.append(sample['file_id'])
    return np.concatenate(voxel_features), np.concatenate(voxel_coords), \
        np.array(pos_equal_one), np.array(neg_equal_one), \
        np.array(targets), images, pointclouds, calibs, ids


class KittiDataset(Dataset):
    """
    [configuration structure]
        training:
            pos_threshold: 0.6
            neg_threshold: 0.45
            class_list: ['Car']
            anchor:
                - w: 1.6
                l: 3.9
                h: 1.56
                yaw: 0.
                - w: 1.6
                l: 3.9
                h: 1.56
                yaw: 90.
        pointcloud:
            range:
                x: [0., 70.4]
                y: [-40., 40.]
                z: [-3., 1.]
            voxel:
                size:
                x: 0.2
                y: 0.2
                z: 0.4
                point_num: 35

    [folder structure]
        - data_path
            - calib
                - *.txt
                    ...
            - image_2
                - *.png
                    ...
            - label_2
                - *.txt
                    ...
            - velodyne
                - *.bin
                    ...
    """

    def __init__(self, data_path, cfg, shuffle=False, aug=False, is_testset=False) -> None:
        self.box_regression_dim = 7

        self.data_path = pathlib.Path(data_path)

        self.pos_threshold = cfg['training']['pos_threshold']
        self.neg_threshold = cfg['training']['neg_threshold']
        self.class_list = cfg['training']['class_list']
        self.anchor_cfg = cfg['training']['anchor']

        self.range_x = cfg['pointcloud']['range']['x']
        self.range_y = cfg['pointcloud']['range']['y']
        self.range_z = cfg['pointcloud']['range']['z']
        self.voxel_size_x = cfg['pointcloud']['voxel']['size']['x']
        self.voxel_size_y = cfg['pointcloud']['voxel']['size']['y']
        self.voxel_size_z = cfg['pointcloud']['voxel']['size']['z']
        self.voxel_pt_num = cfg['pointcloud']['voxel']['point_num']

        self.shuffle = shuffle
        self.aug = aug
        self.is_testset = is_testset

        """
        Generate anchors
        """
        self.Wx = 0
        self.Hy = 0
        self.Dz = 0
        self.anchors_per_position = 0
        self.anchors = self.genAnchors()
        print(f'KittiDataset:: anchor size: {self.anchors.shape}')

        """
        Get pathes of dataset
        """
        self.files_calib = [str(path)
                            for path in self.data_path.glob('calib/*.txt')]
        self.files_image = [str(path)
                            for path in self.data_path.glob('image_2/*.png')]
        self.files_label = [str(path)
                            for path in self.data_path.glob('label_2/*.txt')]
        self.files_lidar = [str(path)
                            for path in self.data_path.glob('velodyne/*.bin')]

        self.files_calib.sort()
        self.files_image.sort()
        self.files_label.sort()
        self.files_lidar.sort()

        # check data
        for i in range(len(self.files_image)):
            # print(int(''.join(filter(str.isdigit, self.files_image[i]))))
            assert (pathlib.Path(self.files_calib[i]).stem == pathlib.Path(
                self.files_image[i]).stem), "KittiDataset:: check data!"
            assert (pathlib.Path(self.files_calib[i]).stem == pathlib.Path(
                self.files_label[i]).stem), "KittiDataset:: check data!"
            assert (pathlib.Path(self.files_calib[i]).stem == pathlib.Path(
                self.files_lidar[i]).stem), "KittiDataset:: check data!"

        nums = len(self.files_calib)
        self.indices = list(range(nums))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        lidar = kitti_utils.getLidarFromPath(self.files_lidar[idx])
        lidar = lidar[lidar[:, 0] >= self.range_x[0], :]
        lidar = lidar[lidar[:, 0] < self.range_y[1], :]
        lidar = lidar[lidar[:, 1] >= self.range_y[0], :]
        lidar = lidar[lidar[:, 1] < self.range_y[1], :]
        lidar = lidar[lidar[:, 2] >= self.range_y[0], :]
        lidar = lidar[lidar[:, 2] < self.range_z[1], :]
        voxel_features, voxel_coords = utils.voxelize(lidar,
                                                      self.range_x,
                                                      self.range_y,
                                                      self.range_z,
                                                      self.voxel_size_x,
                                                      self.voxel_size_y,
                                                      self.voxel_size_z,
                                                      self.voxel_pt_num)
        data = {}
        data['lidar'] = lidar
        data["voxel_features"] = voxel_features
        data["voxel_coords"] = voxel_coords
        data["image"] = kitti_utils.getImageFromPath(self.files_image[idx])
        data["calib"] = kitti_utils.getCalibrationFromPath(
            self.files_calib[idx])
        data["file_id"] = pathlib.Path(self.files_calib[idx]).stem

        if not self.is_testset:
            label = kitti_utils.getLabelFromPath(self.files_label[idx])
            kitti_utils.parseLabelData(label, data["calib"])
            targets, pos_equal_one, neg_equal_one = self.getTarget(label)
            data["label"] = label
            data["pos_equal_one"] = pos_equal_one
            data["neg_equal_one"] = neg_equal_one
            data["target"] = targets
        else:
            # print(456)
            data["label"] = None
            data["pos_equal_one"] = None
            data["neg_equal_one"] = None
            data["target"] = None

        return data

    def genAnchors(self):
        """
        set self.anchors_per_position, self.Wx, self.Hy, self.Dz and 
        return anchors ()
        """
        # anchor generation from configuration parameters
        self.Wx = np.ceil((self.range_x[1] - self.range_x[0]
                           ) / self.voxel_size_x).astype(np.int)
        self.Hy = np.ceil((self.range_y[1] - self.range_y[0]
                           ) / self.voxel_size_y).astype(np.int)
        self.Dz = np.ceil((self.range_z[1] - self.range_z[0]
                           ) / self.voxel_size_z).astype(np.int)
        print(f'KittiDataset:: voxel number: W: {self.Wx}')
        print(f'KittiDataset:: voxel number: H: {self.Hy}')
        print(f'KittiDataset:: voxel number: D: {self.Dz}')

        x = np.linspace(self.range_x[0]+self.voxel_size_x,
                        self.range_x[1], int(self.Wx/2), endpoint=False)
        y = np.linspace(self.range_y[0]+self.voxel_size_x,
                        self.range_y[1], int(self.Hy/2), endpoint=False)
        cx, cy = np.meshgrid(x, y)
        self.anchors_per_position = len(self.anchor_cfg)
        cx = np.tile(cx[..., np.newaxis], self.anchors_per_position)
        cy = np.tile(cy[..., np.newaxis], self.anchors_per_position)
        cz = np.ones_like(cx) * -1.
        w = np.ones_like(cx)
        l = np.ones_like(cx)
        h = np.ones_like(cx)
        yaw = np.ones_like(cx)

        for i, a in enumerate(self.anchor_cfg):
            w[..., i] = a['w']
            l[..., i] = a['l']
            h[..., i] = a['h']
            yaw[..., i] = a['yaw']

        return np.stack([cx, cy, cz, w, l, h, yaw], axis=-1)

    def getTarget(self, label):
        """
        arg: label (N)
        return  targets (W, H, Anchor number, box regression) 
                pos_equal_one (W, H, Anchor number)
                neg_equal_one (W, H, Anchor number)
        """
        targets = np.zeros((*self.anchors.shape[:3], self.box_regression_dim))
        pos_equal_one = np.zeros(self.anchors.shape[:3])
        neg_equal_one = np.zeros(self.anchors.shape[:3])
        _targets = []
        for key in label:
            if key in self.class_list:
                for data in label[key]:
                    _targets.append(
                        [
                            data['x'],
                            data['y'],
                            data['z'],
                            data['w'],
                            data['l'],
                            data['h'],
                            data['yaw']
                        ]
                    )
        _targets = np.array(_targets)
        corners_targets = utils.getBBoxStandupCorners2DBatch(_targets)
        anchors = np.copy(self.anchors)
        corners_anchors = utils.getBBoxStandupCorners2DBatch(
            np.reshape(anchors, (-1, 7)))
        ious = utils.calculateIouJit(corners_targets, corners_anchors)

        # highest anchors of all targets
        idx_highest = np.argmax(ious.T, axis=0)
        idx_highest_label = np.arange(idx_highest.shape[0])

        # positive anchors
        idx_pos_label, idx_pos = np.where(ious > self.pos_threshold)

        # add hightest anchors and remove duplicated anchors
        idx_pos = np.concatenate([idx_pos, idx_highest])
        idx_pos_label = np.concatenate([idx_pos_label, idx_highest_label])
        idx_pos, idx = np.unique(idx_pos, return_index=True)
        idx_pos_label = idx_pos_label[idx]

        # negative anchors
        idx_neg_label, idx_neg = np.where(ious < self.neg_threshold)
        # remove positive anchors
        _, int_idx_neg, _ = np.intersect1d(
            idx_neg, idx_pos, return_indices=True)
        idx_neg = np.delete(idx_neg, int_idx_neg)
        idx_neg_label = np.delete(idx_neg_label, int_idx_neg)

        # TODO: multiple label
        idx_pos_xyza = np.unravel_index(idx_pos, pos_equal_one.shape)
        pos_equal_one[idx_pos_xyza] = 1
        neg_equal_one[np.unravel_index(idx_neg, neg_equal_one.shape)] = 1
        targets[idx_pos_xyza] = _targets[idx_pos_label]

        targets[idx_pos_xyza] = utils.anchorEncoder(
            self.anchors[idx_pos_xyza], targets[idx_pos_xyza])
        return targets, pos_equal_one, neg_equal_one


# if __name__ == "__main__":
#     """
#     for import config, run with module options
#     $python3 -m data.dataset
#     """
#     from config import config as cfg

#     dataset_training = KittiDataset(pathlib.Path(
#         __file__).parent.absolute()/"training", cfg)

#     print("training dataset length: ", len(dataset_training))

#     idx = np.random.choice(len(dataset_training), 1).item()

#     print("training dataset sample {}: ".format(idx), dataset_training[idx])
