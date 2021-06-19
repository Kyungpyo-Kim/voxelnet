import pathlib
import os
import glob
from torch.utils.data import Dataset, dataset
import numpy as np


class KittiDataset(Dataset):
    def __init__(self, data_dir, shuffle=False, aug=False,
                 is_testset=False) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.aug = aug
        self.is_testset = is_testset

        self.f_rgb = glob.glob(os.path.join(self.data_dir, 'image_2', '*.png'))
        self.f_lidar = glob.glob(os.path.join(
            self.data_dir, 'velodyne', '*.bin'))
        self.f_label = glob.glob(os.path.join(
            self.data_dir, 'label_2', '*.txt'))

        self.f_rgb.sort()
        self.f_lidar.sort()
        self.f_label.sort()

        self.data_tag = [name.split('/')[-1].split('.')[-2]
                         for name in self.f_rgb]

        assert len(self.data_tag) != 0, 'Dataset folder is not correct!'
        assert len(self.data_tag) == len(self.f_rgb) == len(
            self.f_lidar), 'Dataset folder is not correct!'

        nums = len(self.f_rgb)
        self.indices = list(range(nums))
        if self.shuffle:
            np.random.shuffle(self.indices)

        # Build a data processor
        # self.proc = Processor(self.data_tag, self.f_rgb, self.f_lidar,
        #                       self.f_label, self.data_dir, self.aug, self.is_testset)

    def __getitem__(self, index):
        # [voxel_features, voxel_coords, pos_equal_one, neg_equal_one, targets, images, calibs, ids]
        print(self.f_rgb[index])
        print(self.f_lidar[index])
        print(self.f_label[index])

        data = {}
        data["voxel_features"] = None
        data["voxel_coords"] = None
        data["pos_equal_one"] = None
        data["neg_equal_one"] = None
        data["target"] = None
        data["image"] = None
        data["calibs"] = None
        data["id"] = None
        return data

    def __len__(self):
        return len(self.indices)


if __name__ == "__main__":
    dataset_training = KittiDataset(pathlib.Path(
        __file__).parent.absolute()/"training")
    print("training dataset length: ", len(dataset_training))
    idx = np.random.choice(len(dataset_training), 1).item()
    print("training dataset sample {}: ".format(idx), dataset_training[idx])
