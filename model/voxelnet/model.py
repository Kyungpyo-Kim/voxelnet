import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from config import config as cfg
import numpy as np
# conv2d + bn + relu


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, activation=True,
                 batch_norm=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x

# conv3d + bn + relu


class Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        # B C D H W
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return F.relu(x, inplace=True)

# Fully Connected Network


class FCN(nn.Module):

    def __init__(self, cin, cout):
        """
        cin: input
        cout: output
        """
        super(FCN, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        # KK is the stacked k across batch
        kk, t, _ = x.shape
        x = self.linear(x.view(kk*t, -1))
        x = F.relu(self.bn(x))
        return x.view(kk, t, -1)

# Voxel Feature Encoding layer


class VFE(nn.Module):

    def __init__(self, cin, cout):
        """
        self.units: half of cout (number of features)
        """
        super(VFE, self).__init__()
        assert cout % 2 == 0
        self.units = cout // 2
        self.fcn = FCN(cin, self.units)

    def forward(self, x, mask):
        pointWiseFeature = self.fcn(x)
        # [VoxelNum, MaxPtsNum, units]

        # localAggrFeature = torch.max(pointWiseFeature, 1, keepdim=True)[
        #     0].repeat(1, cfg.T, 1)
        localAggrFeature = torch.max(pointWiseFeature, 1)[
            0].unsqueeze(1).repeat(1, cfg.T, 1)
        # [VoxelNum, 1*MaxPtsNum, units]

        # pointWiseConcat = torch.cat(
        #     (pointWiseFeature, localAggrFeature), dim=-1)
        pointWiseConcat = torch.cat(
            (pointWiseFeature, localAggrFeature), dim=2)
        # [VoxelNum, MaxPtsNum, 2*units]

        # apply mask
        mask = mask.unsqueeze(2).repeat(1, 1, self.units * 2)
        pointWiseConcat = pointWiseConcat * mask.float()

        return pointWiseConcat

# Stacked Voxel Feature Encoding


class SVFE(nn.Module):

    def __init__(self):
        super(SVFE, self).__init__()
        self.vfe_1 = VFE(7, 32)
        self.vfe_2 = VFE(32, 128)
        self.fcn = FCN(128, 128)

    def forward(self, x):
        # x: [VoxelNum, MaxPtsNum, 7]

        # masking for making sparse tensor
        mask = torch.ne(torch.max(x, 2)[0], 0)

        x = self.vfe_1(x, mask)
        # x: [VoxelNum, MaxPtsNum, 32]

        x = self.vfe_2(x, mask)
        # x: [VoxelNum, MaxPtsNum, 128]

        x = self.fcn(x)
        # x: [VoxelNum, MaxPtsNum, 128]

        # element-wise max pooling
        x = torch.max(x, 1)[0]
        # x: [VoxelNum, 128]

        return x

# Convolutional Middle Layer


class CML(nn.Module):
    def __init__(self):
        super(CML, self).__init__()
        self.conv3d_1 = Conv3d(128, 64, 3, s=(2, 1, 1), p=(1, 1, 1))
        self.conv3d_2 = Conv3d(64, 64, 3, s=(1, 1, 1), p=(0, 1, 1))
        self.conv3d_3 = Conv3d(64, 64, 3, s=(2, 1, 1), p=(1, 1, 1))

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        return x

# Region Proposal Network


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        self.block_1 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_1 += [Conv2d(128, 128, 3, 1, 1) for _ in range(3)]
        self.block_1 = nn.Sequential(*self.block_1)

        self.block_2 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_2 += [Conv2d(128, 128, 3, 1, 1) for _ in range(5)]
        self.block_2 = nn.Sequential(*self.block_2)

        self.block_3 = [Conv2d(128, 256, 3, 2, 1)]
        self.block_3 += [nn.Conv2d(256, 256, 3, 1, 1) for _ in range(5)]
        self.block_3 = nn.Sequential(*self.block_3)

        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(
            256, 256, 4, 4, 0), nn.BatchNorm2d(256))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(
            128, 256, 2, 2, 0), nn.BatchNorm2d(256))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(
            128, 256, 1, 1, 0), nn.BatchNorm2d(256))

        self.score_head = Conv2d(
            768, cfg.anchors_per_position, 1, 1, 0, activation=False, batch_norm=False)
        self.reg_head = Conv2d(
            768, 7 * cfg.anchors_per_position, 1, 1, 0, activation=False, batch_norm=False)

    def forward(self, x):
        x = self.block_1(x)
        x_skip_1 = x
        x = self.block_2(x)
        x_skip_2 = x
        x = self.block_3(x)
        x_0 = self.deconv_1(x)
        x_1 = self.deconv_2(x_skip_2)
        x_2 = self.deconv_3(x_skip_1)
        x = torch.cat((x_0, x_1, x_2), 1)
        return self.score_head(x), self.reg_head(x)


class VoxelNet(nn.Module):

    def __init__(self, cfg, device):
        super(VoxelNet, self).__init__()

        # configuration
        self.batch_size = cfg['training']['batch_size']
        self.range_x = cfg['pointcloud']['range']['x']
        self.range_y = cfg['pointcloud']['range']['y']
        self.range_z = cfg['pointcloud']['range']['z']
        self.voxel_size_x = cfg['pointcloud']['voxel']['size']['x']
        self.voxel_size_y = cfg['pointcloud']['voxel']['size']['y']
        self.voxel_size_z = cfg['pointcloud']['voxel']['size']['z']
        self.voxel_pt_num = cfg['pointcloud']['voxel']['point_num']
        self.Wx = np.ceil((self.range_x[1] - self.range_x[0]
                           ) / self.voxel_size_x).astype(np.int)
        self.Hy = np.ceil((self.range_y[1] - self.range_y[0]
                           ) / self.voxel_size_y).astype(np.int)
        self.Dz = np.ceil((self.range_z[1] - self.range_z[0]
                           ) / self.voxel_size_z).astype(np.int)
        print(f'VoxelNet:: voxel number: W: {self.Wx}')
        print(f'VoxelNet:: voxel number: H: {self.Hy}')
        print(f'VoxelNet:: voxel number: D: {self.Dz}')

        self.device = device

        # layers
        self.svfe = SVFE()
        self.cml = CML()
        self.rpn = RPN()

    def voxel_indexing(self, sparse_features, coords):
        # C
        channel = sparse_features.shape[-1]

        # C B Dz Hy Wx
        dense_feature = Variable(torch.zeros(
            channel, self.batch_size,
            self.Dz, self.Hy, self.Wx, dtype=torch.float32).to(self.device))
        dense_feature[:, coords[:, 0], coords[:, 3],
                      coords[:, 2], coords[:, 1]] = sparse_features.transpose(0, 1)
        # B C D H W
        return dense_feature.transpose(0, 1)

    def forward(self, voxel_features, voxel_coords):
        # feature learning network
        vwfs = self.svfe(voxel_features)
        vwfs = self.voxel_indexing(vwfs, voxel_coords)

        # convolutional middle network
        cml_out = self.cml(vwfs)

        # region proposal network
        # merge the depth and feature dim into one, output probability score map
        # and regression map
        probability_score_map, regression_map = self.rpn(
            cml_out.view(cfg.N, -1, cfg.H, cfg.W))

        return probability_score_map, regression_map


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()


def detection_collate(batch):
    voxel_features = []
    voxel_coords = []
    pos_equal_one = []
    neg_equal_one = []
    targets = []
    images = []
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
        calibs.append(sample['calib'])
        ids.append(sample['file_id'])

        return np.concatenate(voxel_features), np.concatenate(voxel_coords), \
            np.array(pos_equal_one), np.array(neg_equal_one), \
            np.array(targets), images, calibs, ids


if __name__ == "_123_main__":
    import numpy as np
    net = Conv3d(128, 64, 3, s=(2, 1, 1), p=(1, 1, 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.train()
    data = np.zeros((1, 128, 10, 400, 352))
    data = Variable(torch.cuda.FloatTensor(data))
    net(data)

if __name__ == "__main__":
    import pathlib
    import os
    import time
    import numpy as np
    from loss import VoxelNetLoss
    from torch import optim
    from torch.utils import data
    from data.dataset import KittiDataset

    print("main")
    print(torch.__version__)
    print(torch.version.cuda)

    # model
    net = VoxelNet()
    print(net)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    print("\n## Hardware type: {}\n".format(device.type))

    # training mode
    net.train()

    # initialization
    print('Initializing weights...')
    net.apply(weights_init)

    # define optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # loss function
    loss = VoxelNetLoss(alpha=1.5, beta=1)

    print(pathlib.Path(os.getcwd()).absolute()/"data/training")

    # dataset
    print(pathlib.Path(os.getcwd()).absolute()/"data/training")
    dataset = KittiDataset(pathlib.Path(
        os.getcwd()).absolute()/"data/training", cfg)
    data_loader = data.DataLoader(dataset, batch_size=cfg.N, shuffle=True,
                                  num_workers=2, collate_fn=detection_collate,
                                  pin_memory=False)
    # training process
    batch_iterator = None
    epoch_size = len(dataset) // cfg.N
    print('Epoch size', epoch_size)

    for iteration in range(1):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            batch_iterator = iter(data_loader)

        voxel_features, voxel_coords, pos_equal_one, neg_equal_one, targets, images, calibs, ids = next(
            batch_iterator)

        voxel_features = Variable(torch.cuda.FloatTensor(voxel_features))
        pos_equal_one = Variable(torch.cuda.FloatTensor(pos_equal_one))
        neg_equal_one = Variable(torch.cuda.FloatTensor(neg_equal_one))
        targets = Variable(torch.cuda.FloatTensor(targets))

        optimizer.zero_grad()

        t0 = time.time()
        psm, rm = net(voxel_features, voxel_coords)
