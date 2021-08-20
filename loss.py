import torch
import torch.nn as nn
import torch.nn.functional as F

small_addon_for_BCE = 1e-6

class VoxelNetLoss(nn.Module):
    def __init__(self, alpha, beta, huber_beta):
        super(VoxelNetLoss, self).__init__()
        self.smoothl1loss = nn.SmoothL1Loss(reduction='sum', beta=huber_beta)
        self.alpha = alpha
        self.beta = beta

    def forward(self, regression_map, probablitity_score_map, pos_equal_one, neg_equal_one, targets):
        """
        regression_map: B C(14) W H
        probablitity_score_map: B C(2) W H
        pos_equal_one: B W H C(2)
        neg_equal_one: B W H C(2)
        targets: B W H C(14)
        """

        # softmax output for positive anchor and negative anchor
        # B W H C(2)
        p_pos = torch.sigmoid(probablitity_score_map.permute(0, 2, 3, 1))
        
        # B W H C(14)
        regression_map = regression_map.permute(0, 2, 3, 1).contiguous()
        
        # B W H C(2) C(7)
        B = regression_map.size(0)
        W = regression_map.size(1)
        H = regression_map.size(2)
        C = int(regression_map.size(3)/2)
        regression_map = regression_map.view(B, W, H, -1, C)

        # B W H C(2) C(7)
        B = targets.size(0)
        W = targets.size(1)
        H = targets.size(2)
        C = int(targets.size(3)/2)
        targets = targets.view(B, W, H, -1, C)
        
        # B W H C(2)
        # -> B W H C(2) C(1)
        # -> B W H C(2) C(7)
        pos_equal_one_for_reg = pos_equal_one.unsqueeze(
            pos_equal_one.dim()).expand(-1, -1, -1, -1, 7)

        # B W H C(2) C(7)
        rm_pos = regression_map * pos_equal_one_for_reg
        # B W H C(2) C(7)
        targets_pos = targets * pos_equal_one_for_reg

        # B W H C(2)
        cls_pos_loss = -pos_equal_one * torch.log(p_pos + small_addon_for_BCE)
        cls_pos_loss = cls_pos_loss.sum() / pos_equal_one.sum()

        # B W H C(2)
        cls_neg_loss = -neg_equal_one * torch.log(1 - p_pos + small_addon_for_BCE)
        cls_neg_loss = cls_neg_loss.sum() / neg_equal_one.sum()

        reg_loss = self.smoothl1loss(rm_pos, targets_pos)
        reg_loss = reg_loss / pos_equal_one.sum()
        conf_loss = self.alpha * cls_pos_loss + self.beta * cls_neg_loss
        return conf_loss, reg_loss
