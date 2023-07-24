import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_loss


@weighted_loss
def border_dist_loss(ps, rect_p, beta=1.0):
    bt, len = rect_p.shape
    assert bt > 0
    assert len > 0
    total_temp = []
    for ind in range(0, 18, 2):
        p = ps[:, ind:ind + 2]
        temp = []
        for i in range(0, 6, 2):
            # 12, 23, 34
            temp.append(single_d(p, rect_p[:, i:i + 2], rect_p[:, i + 2:i + 4]))
        # 41
        temp.append(single_d(p, rect_p[:, 6:8], rect_p[:, 0:2]))
        # a = torch.tensor(temp)
        a = torch.stack(temp, dim=0)
        # print(a.shape)
        a = torch.min(a, dim=0).values
        # print(a.shape)
        total_temp.append(a)
    loss_bd = torch.stack(total_temp, dim=0).mean(dim=0)
    # print(res)
    # res = res.sum(dim=0)
    # print(res)
    area = torch.abs(bt_pts_area(rect_p))
    loss_bd_norm = loss_bd / torch.clamp(torch.sqrt(area), 1e-12)
    return loss_bd_norm

def bt_pts_area(ps):
    b, n = ps.shape
    last_x = ps[:,-2]
    last_y = ps[:,-1]
    first_x = ps[:,0]
    first_y = ps[:,1]

    res = last_x * first_y - last_y * first_x
    for i in range(0, n - 2, 2):
        res += ps[:,i] * ps[:,i + 3] - ps[:,i + 1] * ps[:,i + 2]
    return res / 2.0
def single_d(p, d1, d2):
    [x1, y1, x2, y2, x3, y3] = p[:, 0], p[:, 1], d1[:, 0], d1[:, 1], d2[:, 0], d2[:, 1]

    # 手滑了?
    # s2 = (x1 * y2 - x1 * y3 + x2 * y3 - x2 * y1 + x3 * y1 - x2 * y2)
    s2 = (x1 * y2 - x1 * y3 + x2 * y3 - x2 * y1 + x3 * y1 - x3 * y2)
    # s_2 = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)
    a = torch.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
    d = torch.abs(s2) / torch.clamp(a, 1e-12)
    return d

@LOSSES.register_module
class BorderDistLoss(nn.Module):

    def __init__(self,  loss_weight=1.0, reduction='mean'):
        super(BorderDistLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if pred.shape[0] == 0 or target.shape[0] == 0:
            return pred.new_zeros([1])
        # weight = weight.unsqueeze(dim=1)#.repeat(1, 4) 
        if weight not in (None, 'none'):
            assert weight.dim() == 1
            if avg_factor is None:
                avg_factor = torch.sum(weight > 0).float().item() + 1e-6
        loss = self.loss_weight * border_dist_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor = avg_factor,
            **kwargs)
        return loss
