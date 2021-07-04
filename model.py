import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from config import config as cfg

# conv2d + bn + relu


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, activation=True, batch_norm=True):
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
        print("conv3d")
        x = self.conv(x)
        print("bn")
        if self.bn is not None:
            x = self.bn(x)
        print("end")
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

        localAggrFeature = torch.max(pointWiseFeature, 1, keepdim=True)[0].repeat(1,cfg.T,1)
        # [VoxelNum, 1*MaxPtsNum, units]

        pointWiseConcat = torch.cat(
            (pointWiseFeature, localAggrFeature), dim=-1)
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
        # 여기 디버깅 하기
        self.conv3d_1 = Conv3d(128, 64, 3, s=(2, 1, 1), p=(1, 1, 1))
        self.conv3d_2 = Conv3d(64, 64, 3, s=(1, 1, 1), p=(0, 1, 1))
        self.conv3d_3 = Conv3d(64, 64, 3, s=(2, 1, 1), p=(1, 1, 1))

    def forward(self, x):
        print("cml: 1")
        x = self.conv3d_1(x)
        print("cml: 2")
        x = self.conv3d_2(x)
        print("cml: 3")
        x = self.conv3d_3(x)
        print("end cml")
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

    def __init__(self):
        super(VoxelNet, self).__init__()
        self.svfe = SVFE()
        self.cml = CML()
        self.rpn = RPN()

    def voxel_indexing(self, sparse_features, coords):
        dim = sparse_features.shape[-1]

        dense_feature = Variable(torch.zeros(
            dim, cfg.N, cfg.D, cfg.H, cfg.W).cuda())

        dense_feature[:, coords[:, 0], coords[:, 1],
                      coords[:, 2], coords[:, 3]] = sparse_features.transpose(0, 1)

        return dense_feature.transpose(0, 1)

    def forward(self, voxel_features, voxel_coords):

        # feature learning network
        vwfs = self.svfe(voxel_features)
        vwfs = self.voxel_indexing(vwfs, voxel_coords)

        # convolutional middle network
        print("vwfs: ", vwfs.shape)
        cml_out = self.cml(vwfs)

        # region proposal network

        # merge the depth and feature dim into one, output probability score map and regression map
        probability_score_map, regression_map = self.rpn(
            cml_out.view(cfg.N, -1, cfg.H, cfg.W))

        return probability_score_map, regression_map
