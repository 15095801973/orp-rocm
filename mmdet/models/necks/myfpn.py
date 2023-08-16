import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from mmdet.ops import ConvModule
from ..registry import NECKS


@NECKS.register_module
class MYFPN(nn.Module):
    """
    Feature Pyramid Network.

    This is an implementation of - Feature Pyramid Networks for Object
    Detection (https://arxiv.org/abs/1612.03144)

    Args:
        in_channels (List[int]):
            number of input channels per scale

        out_channels (int):
            number of output channels (used at each scale)

        num_outs (int):
            number of output scales

        start_level (int):
            index of the first input scale to use as an output scale

        end_level (int, default=-1):
            index of the last input scale to use as an output scale

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print('outputs[{}].shape = {!r}'.format(i, outputs[i].shape))
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(MYFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
        
        self.WG_convs1 = nn.ModuleList()
        self.WG_convs3 = nn.ModuleList()
        self.assign_convs1 = nn.ModuleList()
        my_act_cfg = dict(type='ReLU')
        for i in range(num_outs):
            conv3 = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=my_act_cfg,
                    inplace=True)
            self.WG_convs3.append(conv3)
            conv1 = ConvModule(
                    out_channels,
                    1,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=None,
                    act_cfg=None,
                    inplace=True)
            self.WG_convs1.append(conv1)
            assign_conv1 = ConvModule(
                    out_channels,
                    out_channels,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=None,
                    act_cfg=my_act_cfg,
                    inplace=True)
            self.assign_convs1.append(assign_conv1)

        self.rf_conv1_256_16  = ConvModule(
                    out_channels,
                    16,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=None,
                    act_cfg=my_act_cfg,
                    inplace=True)
        self.rf_conv3_16_16_1  = ConvModule(
                    16,
                    16,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=None,
                    act_cfg=my_act_cfg,
                    inplace=True)
        self.rf_conv3_16_16_2  = ConvModule(
                    16,
                    16,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=None,
                    act_cfg=my_act_cfg,
                    inplace=True)
        self.rf_conv1_16_1  = ConvModule(
                    16,
                    1,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=None,
                    act_cfg=my_act_cfg,
                    inplace=True)
        self.rf_conv1_256_16_2  = ConvModule(
                    out_channels,
                    16,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=None,
                    act_cfg=my_act_cfg,
                    inplace=True)
        self.rf_conv1_16_256  = ConvModule(
                    16,
                    256,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=None,
                    act_cfg=my_act_cfg,
                    inplace=True)
    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        lis = []
        for i in range(self.num_outs):
            pool_res = F.adaptive_avg_pool2d(self.WG_convs3[i](outs[i]), (1,1))
            line_res = self.WG_convs1[i](pool_res)
            lis.append(line_res)
        line = torch.cat(lis, dim=0)
        weights = F.softmax(line, dim=0)

        prev_shape = outs[0].shape[2:]
        fusion_feat = outs[0] * weights[0]
        for i in range(1, self.num_outs):
            mul_weight = outs[i] * weights[i]
            fusion_feat += F.interpolate(
                mul_weight, size=prev_shape, mode='nearest')
            
        b1 = self.rf_conv1_256_16(fusion_feat)
        b2 = self.rf_conv3_16_16_2(self.rf_conv3_16_16_1(b1))
        b3 = self.rf_conv1_16_1(b2)

        a1 = F.adaptive_avg_pool2d(fusion_feat, (1,1))
        a2 = self.rf_conv1_256_16_2(a1)
        a3 = self.rf_conv1_16_256(a2)

        a4, b4 = torch.broadcast_tensors(a3, b3)
        mul_res = a4 * b4
        sig_res = mul_res.sigmoid()
        rf_res = fusion_feat + fusion_feat * sig_res
        new_outs=[]

        for i in range(0, self.num_outs):
            prev_shape = outs[i].shape[2:]
            assign_res = self.assign_convs1[i](rf_res)
            out = outs[i] + F.interpolate(
                assign_res, size=prev_shape, mode='nearest')
            new_outs.append(out)
        return tuple(new_outs)

        return tuple(outs)
        
