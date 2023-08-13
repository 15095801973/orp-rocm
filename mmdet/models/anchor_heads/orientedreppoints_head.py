from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.cnn import constant_init
from mmdet.core import (PointGenerator, multi_apply, multiclass_rnms,
                       levels_to_images)
from mmdet.ops import ConvModule, DeformConv
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob
from mmdet.core.bbox import init_pointset_target, refine_pointset_target
from mmdet.ops.minarearect import minaerarect
from mmdet.ops.chamfer_distance import ChamferDistance2D
from mmdet.models.backbones.swin_transformer import SwinTransformerBlock
from mmdet.models.backbones.lsknet import LSKblock
import math
import matplotlib.pyplot as plt

class MYLSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lsk_conv_spatial0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.lsk_conv_spatial1 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # self.lsk_conv_spatial1 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.lsk_conv1 = nn.Conv2d(dim, dim//2, 1)
        self.lsk_conv2 = nn.Conv2d(dim, dim//2, 1)
        self.lsk_conv3 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(6, 3, 7, padding=3)
        self.lsk_conv_cmix = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):   
        
        lsk_c0 = x
        lsk_c1 = self.lsk_conv_spatial0(lsk_c0)
        lsk_c2 = self.lsk_conv_spatial1(lsk_c1)

        attn0 = self.lsk_conv1(lsk_c0)
        attn1 = self.lsk_conv2(lsk_c1)
        attn2 = self.lsk_conv3(lsk_c2)
        
        # attn = torch.cat([attn1, attn2], dim=1)
        avg_attn0 = torch.mean(attn0, dim=1, keepdim=True)
        max_attn0, _ = torch.max(attn0, dim=1, keepdim=True)
        avg_attn1 = torch.mean(attn1, dim=1, keepdim=True)
        max_attn1, _ = torch.max(attn1, dim=1, keepdim=True)
        avg_attn2 = torch.mean(attn2, dim=1, keepdim=True)
        max_attn2, _ = torch.max(attn2, dim=1, keepdim=True)
        agg = torch.cat([avg_attn0, max_attn0,avg_attn1, max_attn1,avg_attn2, max_attn2], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn_mixd = attn0 * sig[:,0,:,:].unsqueeze(1) + attn1 * sig[:,1,:,:].unsqueeze(1) + attn2 * sig[:,2,:,:].unsqueeze(1)
        lsk_res = self.lsk_conv_cmix(attn_mixd)
        return lsk_res

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

@HEADS.register_module
class OrientedRepPointsHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 point_feat_channels=256,
                 stacked_convs=3,
                 num_points=9,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_rbox_init=dict(
                     type='IoULoss', loss_weight=0.4),
                 loss_rbox_refine=dict(
                     type='IoULoss', loss_weight=0.75),
                 loss_spatial_init=dict(type='SpatialBorderLoss', loss_weight=0.05),
                 loss_spatial_refine=dict(type='SpatialBorderLoss', loss_weight=0.05),
                 center_init=True,
                 top_ratio=0.4,
                 # is_division=True,
                 # is_division_pts=True
                 my_pts_mode="demo", #"demo"
                 loss_border_dist_init = dict(type='BorderDistLoss', loss_weight=0.2),
                 loss_border_dist_refine = dict(type='BorderDistLoss', loss_weight=0.8),
                 attn_drop=0.
                 ):

        super(OrientedRepPointsHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.point_feat_channels = point_feat_channels
        self.stacked_convs = stacked_convs
        self.num_points = num_points
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        # 为True时分类器输出通道数为num_classes - 1
        # 为False时分类器输出通道数为num_classes
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss']
        self.loss_cls = build_loss(loss_cls)
        self.loss_rbox_init = build_loss(loss_rbox_init)
        self.loss_rbox_refine = build_loss(loss_rbox_refine)
        self.loss_spatial_init = build_loss(loss_spatial_init)
        self.loss_spatial_refine = build_loss(loss_spatial_refine)
        # 
        self.loss_border_dist = build_loss(loss_border_dist_init) if  loss_border_dist_init is not None else None
        self.loss_border_dist_refine = build_loss(loss_border_dist_refine) if  loss_border_dist_refine is not None else None
        self.drop = nn.Dropout(0.1)
        self.attn_drop=attn_drop

        self.center_init = center_init
        self.top_ratio = top_ratio
        self.my_pts_mode = my_pts_mode
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes
        self.point_generators = [PointGenerator() for _ in self.point_strides]
        # we use deformable conv to extract points features
        # 我们使用可形变卷积来提取点特征
        # 开方以获取边长度
        self.dcn_kernel = int(np.sqrt(num_points))
        # 计算填充值,保持卷积后尺寸不变
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        # 确保点数是可以开方的
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        # 计算可形变卷积的基础偏移,例如(-1,-1),(-1,0),...,(1,1)
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        # TODO 5x5 DCN base offset
        self.sup_num_points = 81
        self.sup_dcn_kernel = int(np.sqrt(self.sup_num_points))
        # 计算填充值,保持卷积后尺寸不变
        self.sup_dcn_pad = int((self.sup_dcn_kernel - 1) / 2)
        # 确保点数是可以开方的
        assert self.sup_dcn_kernel * self.sup_dcn_kernel == self.sup_num_points, \
            'The points number should be a square number.'
        assert self.sup_dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        # 计算可形变卷积的基础偏移,例如(-1,-1),(-1,0),...,(1,1)
        sup_dcn_base = np.arange(-self.sup_dcn_pad,
                             self.sup_dcn_pad + 1).astype(np.float64)
        sup_dcn_base_y = np.repeat(sup_dcn_base, self.sup_dcn_kernel)
        sup_dcn_base_x = np.tile(sup_dcn_base, self.sup_dcn_kernel)
        sup_dcn_base_offset = np.stack([sup_dcn_base_y, sup_dcn_base_x], axis=1).reshape(
            (-1))
        self.sup_dcn_base_offset = torch.tensor(sup_dcn_base_offset).view(1, -1, 1, 1)
        # ------DID END-----
        
        # 初始化各网络层
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.relu_no_inplace = nn.ReLU(inplace=False)
        # 初始化分类网络层
        self.cls_convs = nn.ModuleList()
        # 初始化回归网络层
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            # 通道数
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        # 一个点有横纵两个坐标,因此维度*2
        pts_out_dim = 2 * self.num_points
        # 可形变卷积,对特征点进行分类
        # TODO
        # self.reppoints_cls_conv = DeformConv(self.feat_channels,
        #                                      self.point_feat_channels,
        #                                      self.dcn_kernel, 1, self.dcn_pad)
        # 与上一层同时起作用
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)
        # 除了添加一个隐藏层没有特别的含义
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        # 以上一层的输出为输入,输出了代表点,也是可形变卷积的xy偏移
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)

        #与下一层一起起微调代表点位置的作用
        self.reppoints_pts_refine_conv = DeformConv(self.feat_channels,
                                                    self.point_feat_channels,
                                                    self.dcn_kernel, 1,
                                                    self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)
        
        self.forward_pts_refine_conv = DeformConv(self.feat_channels,
                                                    self.point_feat_channels,
                                                    self.dcn_kernel, 1,
                                                    self.dcn_pad)
        self.forward_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)
        self.cls_pts_assis_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)
        # TODO 添加自适应点,专门用来分类
        if self.my_pts_mode == "pts_down" or self.my_pts_mode == "pts_up":
            self.div_reppoints_conv = nn.Conv2d(self.feat_channels,
                                                     self.point_feat_channels, 3,
                                                     1, 1)
            # 以上一层的输出为输入,输出了代表点,也是可形变卷积的xy偏移
            self.div_reppoints_point = nn.Conv2d(self.point_feat_channels,
                                                    pts_out_dim, 1, 1, 0)
            self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
        elif self.my_pts_mode == "mix_up":
            # self.div_reppoints_conv = nn.Conv2d(self.feat_channels,
            #                                          self.point_feat_channels, 3,
            #                                          1, 1)
            self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
            # 以上一层的输出为输入,输出了代表点,也是可形变卷积的xy偏移
            # self.div_reppoints_point = nn.Conv2d(self.point_feat_channels,
            #                                         pts_out_dim, 1, 1, 0)
            dim = 256
            self.lsk_conv_spatial0_cls = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
            self.lsk_conv_spatial1_cls = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
            # self.lsk_conv_spatial1 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
            self.lsk_conv1_cls = nn.Conv2d(dim, dim//2, 1)
            self.lsk_conv2_cls = nn.Conv2d(dim, dim//2, 1)
            self.lsk_conv3_cls = nn.Conv2d(dim, dim//2, 1)
            # self.conv_squeeze_cls = nn.Conv2d(6, 6, 7, padding=3)
            self.lsk_conv_cmix_cls0 = nn.Conv2d(dim//2, dim, 1)
            self.lsk_conv_cmix_cls1 = nn.Conv2d(dim//2, dim, 1)

            self.lsk_conv_spatial0_pts = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
            self.lsk_conv_spatial1_pts = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
            # self.lsk_conv_spatial1 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
            self.lsk_conv1_pts = nn.Conv2d(dim, dim//2, 1)
            self.lsk_conv2_pts = nn.Conv2d(dim, dim//2, 1)
            self.lsk_conv3_pts = nn.Conv2d(dim, dim//2, 1)
            # self.conv_squeeze_pts = nn.Conv2d(6, 6, 7, padding=3)
            self.lsk_conv_cmix_pts0 = nn.Conv2d(dim//2, dim, 1)
            self.lsk_conv_cmix_pts1 = nn.Conv2d(dim//2, dim, 1)

            self.conv_squeeze_mix = nn.Conv2d(12, 12, 7, padding=3)

            self.pseudo_dcn_pts = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
            self.pseudo_dcn_cls = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
            
              
        elif self.my_pts_mode == "int":
            self.div_common_conv1 = nn.Conv2d(self.feat_channels,
                                              self.point_feat_channels, 1, 1, 0)
        elif self.my_pts_mode == "com1":
            self.div_common_conv1 = nn.Conv2d(self.feat_channels,
                                              self.point_feat_channels, 1, 1, 0)
        elif self.my_pts_mode == "com3":
            self.div_common_conv1 = nn.Conv2d(self.feat_channels,
                                              self.point_feat_channels, 3, 1, 1)
        elif self.my_pts_mode == "com5":
            self.div_common_conv1 = nn.Conv2d(self.feat_channels,
                                              self.point_feat_channels, 5, 1, 2)
        elif self.my_pts_mode == "demo" or self.my_pts_mode == "pts_down" or \
            self.my_pts_mode == "pts_up" or self.my_pts_mode == "int" or self.my_pts_mode == "drop":
            self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
            # self.sup_conv1 = nn.Conv2d(self.feat_channels,
            #                                   self.point_feat_channels, 3, 1, 1)
            # self.sup_conv2 = nn.Conv2d(self.feat_channels,
            #                                   self.point_feat_channels, 3, 1, 1)
            self.ddim_conv1 = nn.Conv2d(self.feat_channels*2,
                                              self.point_feat_channels, 1, 1, 0)
            self.ddim_conv2 = nn.Conv2d(self.feat_channels*2,
                                              self.point_feat_channels, 1, 1, 0)
        elif self.my_pts_mode == "attn":
            self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
            window_size=7
            mlp_ratio=4.
            qkv_bias=True
            qk_scale=None
            drop=0.
            attn_drop=0.
            drop_path=0.
            norm_layer=nn.LayerNorm
            self.cls_strans = SwinTransformerBlock(
                dim=self.feat_channels,
                num_heads=8,
                window_size=window_size,
                shift_size=0, #0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            self.loc_strans = SwinTransformerBlock(
                dim=self.feat_channels,
                num_heads=8,
                window_size=window_size,
                shift_size=0, #0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            # self.down_dim_conv1 = nn.Conv2d(self.feat_channels,
            #                                   self.point_feat_channels, 1, 1, 0)
        # todo
        # self.reppoints_cls_conv = DeformConv(self.feat_channels,
        #                                      self.point_feat_channels,
        #                                      self.dcn_kernel, 1, self.dcn_pad)
        self.conv1 = nn.Conv2d(self.feat_channels,
                                              self.point_feat_channels, 1, 1, 0)
        self.conv3 = nn.Conv2d(self.feat_channels,
                                              self.point_feat_channels, 3, 1, 1)
        # self.conv5 = nn.Conv2d(self.feat_channels,
                                            #   self.point_feat_channels, 5, 1, 2)
        if self.my_pts_mode == "ide3":
            self.conv_ide3 = nn.Conv2d(self.feat_channels,
                                              self.point_feat_channels, 3, 1, 1)
        if self.my_pts_mode == "ide2":
            self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
        elif self.my_pts_mode == "core":
            # 注意1x1dcn的pad设置为0, 但是1x1源文件中有问题, 还是用伪3x3吧
            # self.core_dcn = DeformConv(self.feat_channels, self.point_feat_channels, 3, 1, 1)
            self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
        elif self.my_pts_mode == "core_v4":
            # 注意1x1dcn的pad设置为0, 但是1x1源文件中有问题, 还是用伪3x3吧
            # self.core_dcn = DeformConv(self.feat_channels, self.point_feat_channels, 3, 1, 1)
            self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
        elif self.my_pts_mode == "sup_dcn":
            self.sup_dcn_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
            # 以上一层的输出为输入,输出了代表点,也是可形变卷积的xy偏移
            self.sup_dcn_out = nn.Conv2d(self.point_feat_channels,
                                                50 - pts_out_dim, 1, 1, 0)
            
            # self.reppoints_cls_conv = DeformConv(self.feat_channels,
            #                                      self.point_feat_channels,
            #                                      self.dcn_kernel, 1, self.dcn_pad)
            # use 5x5 dcn to substitute above 3x3 dcn
            self.sup_dcn = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 5, stride=1, padding=2)
        elif self.my_pts_mode == "core_v2" or self.my_pts_mode == "core_v3":
            # 注意1x1dcn的pad设置为0, 但是1x1源文件中有问题, 还是用伪3x3吧
            # self.core_dcn = DeformConv(self.feat_channels, self.point_feat_channels, 3, 1, 1)
            self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
            self.pseudo_dcn_pts = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
            self.pseudo_dcn_cls = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
            self.ddim_conv1 = nn.Conv2d(self.feat_channels*2,
                                              self.point_feat_channels, 1, 1, 0)
            self.ddim_conv2 = nn.Conv2d(self.feat_channels*2,
                                              self.point_feat_channels, 1, 1, 0)
            
            self.ct_rate_conv1 = nn.Conv2d(self.feat_channels,
                                              self.num_points, 1, 1, 0)
            self.ct_rate_conv2 = nn.Conv2d(self.feat_channels,
                                              self.num_points, 1, 1, 0)
            
            self.ct_dcn_pts = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
            self.ct_dcn_cls = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
        elif self.my_pts_mode == "fusion":
            self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
            dim = 256
            self.l_conv1 = nn.Conv2d(dim, dim, 1)
            self.l_conv2 = nn.Conv2d(dim, dim, 1)
            # self.l_conv_linear = nn.Conv2d(dim, 2, 1)
            self.l_conv_cls_do = nn.Conv2d(dim, dim//2, 1)
            self.l_conv_pts_do = nn.Conv2d(dim, dim//2, 1)
            self.l_conv_cls_up = nn.Conv2d(dim, dim, 1)
            self.l_conv_pts_up = nn.Conv2d(dim, dim, 1)
            self.l_conv_fus_up = nn.Conv2d(dim, dim, 1)
            self.my_lsk_cls = MYLSKblock(256)
            self.my_lsk_pts = MYLSKblock(256)
            self.my_lsk_fusion = MYLSKblock(256)
            self.pseudo_dcn_pts = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
            self.pseudo_dcn_cls = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
            
        elif self.my_pts_mode == "mix_up_v2":
            # self.div_reppoints_conv = nn.Conv2d(self.feat_channels,
            #                                          self.point_feat_channels, 3,
            #                                          1, 1)
            self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
            # 以上一层的输出为输入,输出了代表点,也是可形变卷积的xy偏移
            # self.div_reppoints_point = nn.Conv2d(self.point_feat_channels,
            #                                         pts_out_dim, 1, 1, 0)
            dim = 256
            self.l_conv1 = nn.Conv2d(dim, dim, 1)
            self.l_conv2 = nn.Conv2d(dim, dim, 1)
            # self.l_conv_linear = nn.Conv2d(dim, 2, 1)

            # Re edit by ConvModule
            # self.l_conv_cls_do = nn.Conv2d(dim, dim//2, 1)
            # self.l_conv_pts_do = nn.Conv2d(dim, dim//2, 1)
            # self.l_conv_cls_up = nn.Conv2d(dim, dim, 1)
            # self.l_conv_pts_up = nn.Conv2d(dim, dim, 1)
            # self.l_conv_fus_up = nn.Conv2d(dim, dim, 1)

            self.lsk_conv_spatial0_cls = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
            self.lsk_conv_spatial1_cls = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
            # self.lsk_conv_spatial1 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
            self.lsk_conv1_cls = nn.Conv2d(dim, dim//2, 1)
            self.lsk_conv2_cls = nn.Conv2d(dim, dim//2, 1)
            self.lsk_conv3_cls = nn.Conv2d(dim, dim//2, 1)
            self.conv_squeeze_cls = nn.Conv2d(6, 6, 7, padding=3)
            self.lsk_conv_cmix_cls0 = nn.Conv2d(dim//2, dim, 1)
            self.lsk_conv_cmix_cls1 = nn.Conv2d(dim//2, dim, 1)

            self.lsk_conv_spatial0_pts = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
            self.lsk_conv_spatial1_pts = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
            # self.lsk_conv_spatial1 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
            self.lsk_conv1_pts = nn.Conv2d(dim, dim//2, 1)
            self.lsk_conv2_pts = nn.Conv2d(dim, dim//2, 1)
            self.lsk_conv3_pts = nn.Conv2d(dim, dim//2, 1)
            self.conv_squeeze_pts = nn.Conv2d(6, 6, 7, padding=3)
            self.lsk_conv_cmix_pts0 = nn.Conv2d(dim//2, dim, 1)
            self.lsk_conv_cmix_pts1 = nn.Conv2d(dim//2, dim, 1)

            self.lsk_conv_spatial0_fus = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
            self.lsk_conv_spatial1_fus = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
            # self.lsk_conv_spatial1 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
            self.lsk_conv1_fus = nn.Conv2d(dim, dim//2, 1)
            self.lsk_conv2_fus = nn.Conv2d(dim, dim//2, 1)
            self.lsk_conv3_fus = nn.Conv2d(dim, dim//2, 1)
            self.conv_squeeze_fus = nn.Conv2d(6, 6, 7, padding=3)
            self.lsk_conv_cmix_fus0 = nn.Conv2d(dim//2, dim, 1)
            self.lsk_conv_cmix_fus1 = nn.Conv2d(dim//2, dim, 1)

            # self.conv_squeeze_mix = nn.Conv2d(12, 12, 7, padding=3)

            self.pseudo_dcn_pts = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
            self.pseudo_dcn_cls = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
            
            self.l_conv_cls_do = ConvModule(dim,dim//2,
                    1,stride=1,padding=0,conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg)
            self.l_conv_pts_do = ConvModule(dim,dim//2,
                    1,stride=1,padding=0,conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg)
            self.l_conv_cls_up = ConvModule(dim,dim,
                    1,stride=1,padding=0,conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,act_cfg= None)
            self.l_conv_pts_up = ConvModule(dim,dim,
                    1,stride=1,padding=0,conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,act_cfg= None)
            self.l_conv_fus_up = ConvModule(dim,dim,
                    1,stride=1,padding=0,conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,act_cfg= None)
           
              
        # self.attn_drop = nn.Dropout(self.attn_drop)
        # self.softmax = nn.Softmax(dim=-1)    
        # self.ape = nn.Linear(2,self.point_feat_channels , bias=True)
        # # self.ape = nn.Embedding(2,self.point_feat_channels)
        # nn.init.constant_(self.ape.weight, 0)
        # # self.norm = norm_layer(2)
        # self.qkv = nn.Linear(256, 256 * 3, bias=True)
        # self.proj = nn.Linear(256, 256)
        # self.proj_drop = nn.Dropout(0.1)
        self.spatial_gating_unit = LSKblock(256)
        self.offset_encoder = nn.Conv2d(256+6, 256, 1)
        self.offset_encoder2 = nn.Conv2d(256, 256, 1)
        self.offset_encoder3 = nn.Conv2d(6, 256, 1)
        self.sele_2 = nn.Conv2d(256, 4, 1)
        self.sele_1 = nn.Conv2d(256, 256, 1)
        self.conv_cat = nn.Conv2d(512, 256, 1)
        self.decomp_conv1_cls = nn.Conv2d(256*3, 256*3, 1)
        self.decomp_conv_d_cls = nn.Conv2d(256*3, 256, 1)
        self.decomp_conv_spatial_cls = nn.Conv2d(6 , 3, 7, padding=3)
        self.decomp_conv1_pts = nn.Conv2d(256*3, 256*3, 1)
        self.decomp_conv_d_pts = nn.Conv2d(256*3, 256, 1)
        self.decomp_conv_spatial_pts = nn.Conv2d(6 , 3, 7, padding=3)
        self.pseudo_dcn_cls = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
            
        # self.sup_dcn_out_points = nn.Conv2d(self.point_feat_channels,
        #                                         162, 1, 1, 0)
        
        # self.sup_dcn_out_points_head1 = nn.Conv2d(self.point_feat_channels,
        #                                         18, 1, 1, 0)
        # self.sup_dcn_out_points_head2 = nn.Conv2d(self.point_feat_channels,
        #                                         18, 1, 1, 0)
        # self.sup_dcn_out_points_head3 = nn.Conv2d(self.point_feat_channels,
        #                                         18, 1, 1, 0)
        # self.sup_dcn = DeformConv(self.feat_channels,
        #                                          self.point_feat_channels,
        #                                          9, stride=1, padding=4)
        # dim = 256
        # self.lsk_conv_spatial0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # self.lsk_conv_spatial1 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # # # self.lsk_conv_spatial1 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # self.l_conv1 = nn.Conv2d(dim, dim, 1)
        # self.l_conv2 = nn.Conv2d(dim, dim, 1)
        # self.l_conv3 = nn.Conv2d(dim, dim, 1)
        # self.l_conv_linear = nn.Conv2d(dim, 2*3, 1)
        # self.l_conv_cls_do = nn.Conv2d(dim, dim//2, 1)
        # self.l_conv_cls_up = nn.Conv2d(dim//2, dim, 1)
        # self.l_conv_pts_do = nn.Conv2d(dim, dim//2, 1)
        # self.l_conv_pts_up = nn.Conv2d(dim//2, dim, 1)
        # # self.conv_squeeze = nn.Conv2d(6, 3, 7, padding=3)
        # # self.lsk_conv_cmix = nn.Conv2d(dim//2, dim, 1)
        # self.my_lsk_cls = MYLSKblock(256)
        # self.my_lsk_pts = MYLSKblock(256)
        # self.my_lsk_fusion = MYLSKblock(256)
        # self.conv_spatial0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # self.conv_spatial1 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

    def init_weights(self):
        # 用标准分布初始化网络层权重
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
            
        bias_cls = bias_init_with_prob(0.01)
        #
        # normal_init(self.reppoints_cls_conv, std=0.01)
        normal_init(self.reppoints_cls_out, std=0.01, bias=bias_cls)
        normal_init(self.reppoints_pts_init_conv, std=0.01)
        normal_init(self.reppoints_pts_init_out, std=0.01)
        normal_init(self.reppoints_pts_refine_conv, std=0.01)
        normal_init(self.reppoints_pts_refine_out, std=0.01)
        # copy
        if self.my_pts_mode == "pts_down" or self.my_pts_mode == "pts_up":
            normal_init(self.div_reppoints_conv, std=0.01)
            normal_init(self.div_reppoints_point, std=0.01)
        # elif self.my_pts_mode == "com3" or self.my_pts_mode == "com1":
        elif self.my_pts_mode[0:3] == "com":
                normal_init(self.div_common_conv1, std=0.01)
        elif self.my_pts_mode == "int":
                normal_init(self.div_common_conv1, std=0.01)
        # else:
        elif self.my_pts_mode == "demo" or self.my_pts_mode == "pts_down" or self.my_pts_mode == "pts_up" or self.my_pts_mode == "int" or self.my_pts_mode == "drop":
            normal_init(self.reppoints_cls_conv, std=0.01)
        elif self.my_pts_mode == "attn":
            normal_init(self.reppoints_cls_conv, std=0.01)

        elif self.my_pts_mode == "ide3":
            normal_init(self.conv_ide3, std=0.01)
        
        elif self.my_pts_mode == "sup_dcn":
            normal_init(self.sup_dcn, std=0.01)
            normal_init(self.sup_dcn_conv, std=0.01)
            normal_init(self.sup_dcn_out, std=0.01)

        elif self.my_pts_mode == "core_v4":
            normal_init(self.reppoints_cls_conv, std=0.01)
        elif self.my_pts_mode == "core":
            normal_init(self.reppoints_cls_conv, std=0.01)
        elif self.my_pts_mode == "core_v2" or self.my_pts_mode == "core_v3":
            normal_init(self.reppoints_cls_conv, std=0.01)
            normal_init(self.pseudo_dcn_pts, std=0.01)
            normal_init(self.pseudo_dcn_cls, std=0.01)
            normal_init(self.ddim_conv1, std=0.01)
            normal_init(self.ddim_conv2, std=0.01)
            normal_init(self.ct_rate_conv1, std=0.01)
            normal_init(self.ct_rate_conv2, std=0.01)
            normal_init(self.ct_dcn_cls, std=0.01)
            normal_init(self.ct_dcn_pts, std=0.01)
        normal_init(self.conv1, std=0.01)
        # normal_init(self.conv3, std=0.01)
        # todo
        # normal_init(self.reppoints_cls_conv, std=0.01)
        for m in self.spatial_gating_unit.modules():
                # if isinstance(m, nn.Linear):
                    # trunc_normal_init(m, std=.02, bias=0.)
                if isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
    # end

    def forward_single(self, x):
        """
        单次向前传播的方法
        Args:
            x: 输入数据
        Returns:cls_out, pts_out_init, pts_out_refine, x
        """

        # 确保为Tensor类型
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        points_init = 0
        # 分类和回归共用初始x
        cls_feat = x
        pts_feat = x
        DECOMP = False
        DECOMP = True
        if DECOMP:
            # ------------cls decomp part-------------
            cls_feat1 = self.cls_convs[0](cls_feat)
            cls_feat2 = self.cls_convs[1](cls_feat1)
            cls_feat3 = self.cls_convs[2](cls_feat2)
            lis1 = torch.cat([cls_feat1, cls_feat2,cls_feat3], dim = 1)
            # spatial pooling
            lis1_avg = F.adaptive_avg_pool2d(lis1, (1,1))
            cannel_attn = self.decomp_conv1_cls(lis1_avg)

            # cannel pooling
            cls_feat1_max, _ = torch.max(cls_feat1, dim=1, keepdim=True)
            cls_feat1_mean = torch.mean(cls_feat1, dim=1, keepdim=True)
            cls_feat2_max, _ = torch.max(cls_feat2, dim=1, keepdim=True)
            cls_feat2_mean = torch.mean(cls_feat2, dim=1, keepdim=True)
            cls_feat3_max, _ = torch.max(cls_feat3, dim=1, keepdim=True)
            cls_feat3_mean = torch.mean(cls_feat3, dim=1, keepdim=True)
            spatial_attn = self.decomp_conv_spatial_cls(torch.cat([cls_feat1_max, cls_feat1_mean,cls_feat2_max, cls_feat2_mean,cls_feat3_max, cls_feat3_mean], dim=1))
            # cls_feat1_attned =  cls_feat1 *spatial_attn[:,0:1].sigmoid() *2
            # cls_feat2_attned =  cls_feat2 *spatial_attn[:,1:2].sigmoid() *2
            # cls_feat3_attned =  cls_feat3 *spatial_attn[:,2:3].sigmoid() *2
            b = spatial_attn.repeat_interleave(256,1)
            a2, b2 = torch.broadcast_tensors(cannel_attn, b)
            mul_res= a2 *b2
            lis2 = lis1 * mul_res.sigmoid()
            # lis2 = torch.cat([cls_feat1_attned, cls_feat2_attned, cls_feat3_attned], dim = 1) 
            # lis2 = torch.cat([cls_feat1_attned, cls_feat2_attned, cls_feat3_attned], dim = 1) *2* cannel_attn.sigmoid()
            cls_feat = self.decomp_conv_d_cls(lis2)
            # --------pts decomp part---------
            pts_feat1 = self.reg_convs[0](pts_feat)
            pts_feat2 = self.reg_convs[1](pts_feat1)
            pts_feat3 = self.reg_convs[2](pts_feat2)
            lis1_pts = torch.cat([pts_feat1, pts_feat2,pts_feat3], dim = 1)
            # spatial pooling
            lis1_avg_pts = F.adaptive_avg_pool2d(lis1_pts, (1,1))
            cannel_attn_pts = self.decomp_conv1_pts(lis1_avg_pts)

            # cannel pooling
            pts_feat1_max, _ = torch.max(pts_feat1, dim=1, keepdim=True)
            pts_feat1_mean = torch.mean(pts_feat1, dim=1, keepdim=True)
            pts_feat2_max, _ = torch.max(pts_feat2, dim=1, keepdim=True)
            pts_feat2_mean = torch.mean(pts_feat2, dim=1, keepdim=True)
            pts_feat3_max, _ = torch.max(pts_feat3, dim=1, keepdim=True)
            pts_feat3_mean = torch.mean(pts_feat3, dim=1, keepdim=True)
            spatial_attn_pts = self.decomp_conv_spatial_pts(torch.cat([pts_feat1_max, pts_feat1_mean,pts_feat2_max, pts_feat2_mean,pts_feat3_max, pts_feat3_mean], dim=1))
            # pts_feat1_attned =  pts_feat1 *spatial_attn_pts[:,0:1].sigmoid() *2
            # pts_feat2_attned =  pts_feat2 *spatial_attn_pts[:,1:2].sigmoid() *2
            # pts_feat3_attned =  pts_feat3 *spatial_attn_pts[:,2:3].sigmoid() *2
            b_pts = spatial_attn_pts.repeat_interleave(256,1)
            a2_pts, b2_pts = torch.broadcast_tensors(cannel_attn_pts, b_pts)
            mul_res_pts= a2_pts *b2_pts
            lis2_pts = lis1_pts * mul_res_pts.sigmoid()
            # lis2_pts = torch.cat([pts_feat1_attned, pts_feat2_attned, pts_feat3_attned], dim = 1) 
            # lis2_pts = torch.cat([pts_feat1_attned, pts_feat2_attned, pts_feat3_attned], dim = 1) *2* cannel_attn_pts.sigmoid()
            pts_feat = self.decomp_conv_d_pts(lis2_pts)
        else:
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                pts_feat = reg_conv(pts_feat)
        # TODO 
        # Fusion
        if self.my_pts_mode == "fusion":  #0803
            # cls_code = self.l_conv_cls_do(cls_feat)

            # pts_code = self.l_conv_pts_do(pts_feat)
            # fus_code = torch.cat([cls_code,pts_code], dim=1)

            # cls_decode = self.l_conv_cls_up(fus_code)
            # pts_decode = self.l_conv_pts_up(fus_code)
            # fus_decode = self.l_conv_fus_up(fus_code)
            # # fusion_weight = self.l_conv_linear(self.relu(torch.cat([cls_code,pts_code], dim=1))).sigmoid()
            # fusion_feat = cls_decode * fusion_weight[:,0] + pts_decode * fusion_weight[:,1]
            # fusion_cls_feat = cls_decode * fusion_weight[:,2] + pts_decode * fusion_weight[:,3]
            # fusion_pts_feat = cls_decode * fusion_weight[:,4] + pts_decode * fusion_weight[:,5]
            fusion_feat = pts_feat
            fusion_cls_feat = cls_feat
            fusion_pts_feat = pts_feat
            # fusion_cls_feat = self.relu(fusion_cls_feat)
            # fusion_pts_feat = self.relu(fusion_pts_feat)
            # fusion_feat = self.relu(fusion_feat)
            # Ct
            lsk_cls_feat = fusion_cls_feat + fusion_cls_feat * self.my_lsk_cls(fusion_cls_feat)
            lsk_pts_feat = fusion_pts_feat + fusion_pts_feat * self.my_lsk_pts(fusion_pts_feat)
            lsk_fusion_feat = fusion_feat + fusion_feat * self.my_lsk_fusion(fusion_feat)

            pts_out_init = self.reppoints_pts_init_out(self.relu(self.reppoints_pts_init_conv(lsk_fusion_feat)))
            pts_out_init = pts_out_init + points_init

            pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
            dcn_offset = pts_out_init_grad_mul - dcn_base_offset

           
            if False:
                # if True:
                plt.imshow(cls_feat[0, 0, :, :].cpu().detach().numpy())
                plt.title("cls_feat")
                plt.show()

            # -----end test--------
            dcn_cls_feat = self.reppoints_cls_conv(lsk_cls_feat, dcn_offset) + self.l_conv1(lsk_cls_feat)  #+ self.pseudo_dcn_cls(lsk_cls_res1, core_offset) 
            dcn_pts_feat = self.reppoints_pts_refine_conv(lsk_pts_feat, dcn_offset) + self.l_conv2(lsk_pts_feat)  #+ self.pseudo_dcn_pts(lsk_pts_res1, core_offset)

            # 然后继续卷积,目标是分类
            cls_out = self.reppoints_cls_out(self.relu(dcn_cls_feat))
            # 以及,对代表点位置进行进一步微调,reppoints_pts_refine_conv是可形变卷积
            pts_out_refine = self.reppoints_pts_refine_out(self.relu(dcn_pts_feat))
            # 微调的结果加上基础值
            pts_out_refine = pts_out_refine + pts_out_init.detach()
            return cls_out, pts_out_init, pts_out_refine, x
        
        elif self.my_pts_mode == "mix_up_v2":  # 0804

            # FUSION
            cls_code = self.l_conv_cls_do(cls_feat)

            pts_code = self.l_conv_pts_do(pts_feat)
            fus_code = torch.cat([cls_code,pts_code], dim=1)

            cls_decode = self.l_conv_cls_up(fus_code)
            pts_decode = self.l_conv_pts_up(fus_code)
            fus_decode = self.l_conv_fus_up(fus_code)
            # cls_decode = self.l_conv_cls_up(self.relu(fus_code))
            # pts_decode = self.l_conv_pts_up(self.relu(fus_code))
            # fus_decode = self.l_conv_fus_up(self.relu(fus_code))
            # fusion_weight = self.l_conv_linear(self.relu(torch.cat([cls_code,pts_code], dim=1))).sigmoid()
            # fusion_feat = cls_decode * fusion_weight[:,0] + pts_decode * fusion_weight[:,1]
            # fusion_cls_feat = cls_decode * fusion_weight[:,2] + pts_decode * fusion_weight[:,3]
            # fusion_pts_feat = cls_decode * fusion_weight[:,4] + pts_decode * fusion_weight[:,5]
            fusion_feat = pts_feat + fus_decode
            fusion_cls_feat = cls_feat + cls_decode
            fusion_pts_feat = pts_feat + pts_decode
            fusion_feat = self.relu(fusion_feat)
            fusion_cls_feat = self.relu(fusion_cls_feat)
            fusion_pts_feat = self.relu(fusion_pts_feat)
            # Context
            B, C, W, H = x.shape

            lsk_c0_cls = fusion_cls_feat
            lsk_c1_cls = self.lsk_conv_spatial0_cls(lsk_c0_cls)
            lsk_c2_cls = self.lsk_conv_spatial1_cls(lsk_c1_cls)

            lsk_c0_pts = fusion_pts_feat
            lsk_c1_pts = self.lsk_conv_spatial0_pts(lsk_c0_pts)
            lsk_c2_pts = self.lsk_conv_spatial1_pts(lsk_c1_pts)

            lsk_c0_fus = fusion_feat
            lsk_c1_fus = self.lsk_conv_spatial0_fus(lsk_c0_fus)
            lsk_c2_fus = self.lsk_conv_spatial1_fus(lsk_c1_fus)

            attn0_cls = self.lsk_conv1_cls(lsk_c0_cls)
            attn1_cls = self.lsk_conv2_cls(lsk_c1_cls)
            attn2_cls = self.lsk_conv3_cls(lsk_c2_cls)

            attn0_pts = self.lsk_conv1_pts(lsk_c0_pts)
            attn1_pts = self.lsk_conv2_pts(lsk_c1_pts)
            attn2_pts = self.lsk_conv3_pts(lsk_c2_pts)

            attn0_fus = self.lsk_conv1_fus(lsk_c0_fus)
            attn1_fus = self.lsk_conv2_fus(lsk_c1_fus)
            attn2_fus = self.lsk_conv3_fus(lsk_c2_fus)
            # --------

            ## FEATURE
            # B C 3 W H
            # 必须分开求最大值和平均值，否则会梯度爆炸, unknow reason
            # attn_cls = torch.stack([attn0_cls,attn1_cls,attn2_cls], dim = 2)
            avg_attn0 = torch.mean(attn0_cls, dim=1, keepdim=True)
            max_attn0, _ = torch.max(attn0_cls, dim=1, keepdim=True)
            avg_attn1 = torch.mean(attn1_cls, dim=1, keepdim=True)
            max_attn1, _ = torch.max(attn1_cls, dim=1, keepdim=True)
            avg_attn2 = torch.mean(attn2_cls, dim=1, keepdim=True)
            max_attn2, _ = torch.max(attn2_cls, dim=1, keepdim=True)

            avg_attn0_pts = torch.mean(attn0_pts, dim=1, keepdim=True)
            max_attn0_pts, _ = torch.max(attn0_pts, dim=1, keepdim=True)
            avg_attn1_pts = torch.mean(attn1_pts, dim=1, keepdim=True)
            max_attn1_pts, _ = torch.max(attn1_pts, dim=1, keepdim=True)
            avg_attn2_pts = torch.mean(attn2_pts, dim=1, keepdim=True)
            max_attn2_pts, _ = torch.max(attn2_pts, dim=1, keepdim=True)


            avg_attn0_fus = torch.mean(attn0_fus, dim=1, keepdim=True)
            max_attn0_fus, _ = torch.max(attn0_fus, dim=1, keepdim=True)
            avg_attn1_fus = torch.mean(attn1_fus, dim=1, keepdim=True)
            max_attn1_fus, _ = torch.max(attn1_fus, dim=1, keepdim=True)
            avg_attn2_fus = torch.mean(attn2_fus, dim=1, keepdim=True)
            max_attn2_fus, _ = torch.max(attn2_fus, dim=1, keepdim=True)
            # agg_cls = torch.cat([avg_attn_cls[:,0:1], max_attn_cls[:,0:1],avg_attn_cls[:,1:2], max_attn_cls[:,1:2],avg_attn_cls[:,2:3], max_attn_cls[:,2:3]], dim=1)
            agg_cls = torch.cat([avg_attn0, max_attn0,avg_attn1, max_attn1,avg_attn2, max_attn2], dim=1)
            agg_pts = torch.cat([avg_attn0_pts, max_attn0_pts,avg_attn1_pts, max_attn1_pts,avg_attn2_pts, max_attn2_pts], dim=1)
            agg_fus = torch.cat([avg_attn0_fus, max_attn0_fus,avg_attn1_fus, max_attn1_fus,avg_attn2_fus, max_attn2_fus], dim=1)

            # agg_pts = torch.cat([avg_attn_pts, max_attn_pts], dim=1)
            # B 9 W H
            sig_cls = self.conv_squeeze_cls(agg_cls).sigmoid()
            sig_pts = self.conv_squeeze_pts(agg_pts).sigmoid()
            sig_fus = self.conv_squeeze_fus(agg_fus).sigmoid()
            # r = self.conv_squeeze_mix(torch.cat([agg_cls,agg_pts], dim=1)).sogmoid()
            # sig_cls = r[:,0:6]
            # sig_pts = r[:,6:12]
            # sig_fus = r[:,6:12]

            attn_mixd_cls0 = attn0_cls * sig_cls[:,0,:,:].unsqueeze(1) + attn1_cls * sig_cls[:,1,:,:].unsqueeze(1) + attn2_cls * sig_cls[:,2,:,:].unsqueeze(1)
            attn_mixd_cls1 = attn0_cls * sig_cls[:,3,:,:].unsqueeze(1) + attn1_cls * sig_cls[:,4,:,:].unsqueeze(1) + attn2_cls * sig_cls[:,5,:,:].unsqueeze(1)

            attn_mixd_pts0 = attn0_pts * sig_pts[:,0,:,:].unsqueeze(1) + attn1_pts * sig_pts[:,1,:,:].unsqueeze(1) + attn2_pts * sig_pts[:,2,:,:].unsqueeze(1)
            attn_mixd_pts1 = attn0_pts * sig_pts[:,3,:,:].unsqueeze(1) + attn1_pts * sig_pts[:,4,:,:].unsqueeze(1) + attn2_pts * sig_pts[:,5,:,:].unsqueeze(1)

            attn_mixd_fus0 = attn0_fus * sig_fus[:,0,:,:].unsqueeze(1) + attn1_fus * sig_fus[:,1,:,:].unsqueeze(1) + attn2_fus * sig_fus[:,2,:,:].unsqueeze(1)
            attn_mixd_fus1 = attn0_fus * sig_fus[:,3,:,:].unsqueeze(1) + attn1_fus * sig_fus[:,4,:,:].unsqueeze(1) + attn2_fus * sig_fus[:,5,:,:].unsqueeze(1)
           
            lsk_cls_res0 = self.lsk_conv_cmix_cls0(attn_mixd_cls0)
            lsk_cls_res1 = self.lsk_conv_cmix_cls1(attn_mixd_cls1)


            lsk_pts_res0 = self.lsk_conv_cmix_pts0(attn_mixd_pts0)
            lsk_pts_res1 = self.lsk_conv_cmix_pts1(attn_mixd_pts1)

            lsk_fus_res0 = self.lsk_conv_cmix_fus0(attn_mixd_fus0)
            # lsk_fus_res1 = self.lsk_conv_cmix_fus1(attn_mixd_fus1)

            # assign LSK result
            lsk_out_cls0 = self.relu(fusion_cls_feat + fusion_cls_feat * lsk_cls_res0)
            lsk_out_cls1 = self.relu(fusion_cls_feat + fusion_cls_feat * lsk_cls_res1)
            lsk_out_pts0 = self.relu(fusion_pts_feat + fusion_pts_feat * lsk_pts_res0)
            lsk_out_pts1 = self.relu(fusion_pts_feat + fusion_pts_feat * lsk_pts_res1)
            lsk_out_fus0 = self.relu(fusion_feat + fusion_feat * lsk_fus_res0)
            #--------------------------
            pts_out_init = self.reppoints_pts_init_out(self.relu(self.reppoints_pts_init_conv(lsk_out_fus0)))
            pts_out_init = pts_out_init + points_init
            pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
            dcn_offset = pts_out_init_grad_mul - dcn_base_offset

            pts_x_mean = pts_out_init[:, 0::2].clone()#.mean(dim=1 )#.unsqueeze(1)
            pts_y_mean = pts_out_init[:, 1::2].clone()#.mean(dim=1)#.unsqueeze(1)
            pts_x_mean = torch.mean(pts_x_mean, dim=1 , keepdim=True)
            pts_y_mean = torch.mean(pts_y_mean, dim=1 , keepdim=True)
            core_pts = torch.cat([pts_x_mean, pts_y_mean], dim=1)
            core_pts = core_pts.repeat(1, 9, 1, 1)
            pts_grad_temp = (1 - self.gradient_mul) * core_pts.detach() + self.gradient_mul * core_pts
            core_offset = pts_grad_temp - dcn_base_offset
          
            # if False:
            #     # if True:

            #     plt.imshow(cls_feat[0, 0, :, :].cpu().detach().numpy())
            #     plt.title("cls_feat")
            #     plt.show()

            # -----end test--------
            # dcn_cls_feat = self.reppoints_cls_conv(lsk_out_cls0, dcn_offset) + self.l_conv1(lsk_out_cls1)  #+ self.pseudo_dcn_cls(lsk_cls_res1, core_offset) 
            # dcn_pts_feat = self.reppoints_pts_refine_conv(lsk_out_pts0, dcn_offset) + self.l_conv2(lsk_out_pts1)  #+ self.pseudo_dcn_pts(lsk_pts_res1, core_offset)
            dcn_cls_feat = self.reppoints_cls_conv(lsk_out_cls0, dcn_offset) + self.pseudo_dcn_cls(lsk_out_cls1, core_offset) 
            dcn_pts_feat = self.reppoints_pts_refine_conv(lsk_out_pts0, dcn_offset) + self.pseudo_dcn_pts(lsk_out_pts1, core_offset)

            cls_out = self.reppoints_cls_out(self.relu(dcn_cls_feat))
            pts_out_refine = self.reppoints_pts_refine_out(self.relu(dcn_pts_feat))
            pts_out_refine = pts_out_refine + pts_out_init.detach()
            return cls_out, pts_out_init, pts_out_refine, x
        
        # pts_feat = self.relu(self.sup_conv2(self.relu(self.sup_conv1(pts_feat))))
        # 初始化代表点
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))
        pts_out_init = pts_out_init + points_init
        # copy
        # is_division_pts = False
        # is_division_pts = True
        if self.my_pts_mode == "pts_down":
            # 记录一下,应该是失误,pts_feat和cls_feat连错了,不过似乎也是可行的
            # pts_div = self.div_reppoints_point(
            #     self.relu(self.div_reppoints_conv(pts_feat)))
            pts_div = self.div_reppoints_point(
                self.relu(self.div_reppoints_conv(cls_feat)))
            pts_div = pts_div + points_init
        elif self.my_pts_mode == "pts_up":
            pts_div = self.div_reppoints_point(
                self.relu(self.div_reppoints_conv(pts_feat)))
            pts_div = pts_div #+ points_init
        elif self.my_pts_mode == "drop":
            # torch.autograd.set_detect_anomaly(True)
            # pts_x_mean = pts_out_init[:, 0::2].mean(dim=1, keepdim = True)
            # pts_y_mean = pts_out_init[:, 1::2].mean(dim=1, keepdim = True)
            pts_x_mean = pts_out_init[:, 0::2].clone()#.mean(dim=1 )#.unsqueeze(1)
            pts_y_mean = pts_out_init[:, 1::2].clone()#.mean(dim=1)#.unsqueeze(1)
            pts_x_mean = torch.mean(pts_x_mean, dim=1 , keepdim=True)
            pts_y_mean = torch.mean(pts_y_mean, dim=1 , keepdim=True)
            core_pts = torch.cat([pts_x_mean, pts_y_mean], dim=1)
            #.repeat(1, 9, 1, 1)
            drop_inds = torch.randint(0, pts_out_init.shape[1]//2,[1])
            pts_out_init[:,drop_inds*2:drop_inds*2+2,:,:] = core_pts

        
        
        elif self.my_pts_mode == "mix_up":
            
            B, C, W, H = x.shape

            lsk_c0_cls = cls_feat
            lsk_c1_cls = self.lsk_conv_spatial0_cls(lsk_c0_cls)
            lsk_c2_cls = self.lsk_conv_spatial1_cls(lsk_c1_cls)
            lsk_c0_pts = pts_feat
            lsk_c1_pts = self.lsk_conv_spatial0_pts(lsk_c0_pts)
            lsk_c2_pts = self.lsk_conv_spatial1_pts(lsk_c1_pts)

            attn0_cls = self.lsk_conv1_cls(lsk_c0_cls)
            attn1_cls = self.lsk_conv2_cls(lsk_c1_cls)
            attn2_cls = self.lsk_conv3_cls(lsk_c2_cls)
            attn0_pts = self.lsk_conv1_pts(lsk_c0_pts)
            attn1_pts = self.lsk_conv2_pts(lsk_c1_pts)
            attn2_pts = self.lsk_conv3_pts(lsk_c2_pts)
            # --------

            # B C 3 W H
            # 必须分开求最大值和平均值，否则会梯度爆炸, unknow reason
            # attn_cls = torch.stack([attn0_cls,attn1_cls,attn2_cls], dim = 2)
            avg_attn0 = torch.mean(attn0_cls, dim=1, keepdim=True)
            max_attn0, _ = torch.max(attn0_cls, dim=1, keepdim=True)
            avg_attn1 = torch.mean(attn1_cls, dim=1, keepdim=True)
            max_attn1, _ = torch.max(attn1_cls, dim=1, keepdim=True)
            avg_attn2 = torch.mean(attn2_cls, dim=1, keepdim=True)
            max_attn2, _ = torch.max(attn2_cls, dim=1, keepdim=True)

            avg_attn0_pts = torch.mean(attn0_pts, dim=1, keepdim=True)
            max_attn0_pts, _ = torch.max(attn0_pts, dim=1, keepdim=True)
            avg_attn1_pts = torch.mean(attn1_pts, dim=1, keepdim=True)
            max_attn1_pts, _ = torch.max(attn1_pts, dim=1, keepdim=True)
            avg_attn2_pts = torch.mean(attn2_pts, dim=1, keepdim=True)
            max_attn2_pts, _ = torch.max(attn2_pts, dim=1, keepdim=True)
            # attn_pts = torch.stack([attn0_pts,attn1_pts,attn2_pts], dim = 2)
            # B 3 W H
            # avg_attn_cls = torch.mean(attn_cls, dim=1, keepdim=False)
            # max_attn_cls, _ = torch.max(attn_cls, dim=1, keepdim=False)
            # avg_attn_pts = torch.mean(attn_pts, dim=1, keepdim=False)
            # max_attn_pts, _ = torch.max(attn_pts, dim=1, keepdim=False)
            # B 6 W H
            # agg_cls = torch.cat([avg_attn_cls[:,0:1], max_attn_cls[:,0:1],avg_attn_cls[:,1:2], max_attn_cls[:,1:2],avg_attn_cls[:,2:3], max_attn_cls[:,2:3]], dim=1)
            agg_cls = torch.cat([avg_attn0, max_attn0,avg_attn1, max_attn1,avg_attn2, max_attn2], dim=1)
            agg_pts = torch.cat([avg_attn0_pts, max_attn0_pts,avg_attn1_pts, max_attn1_pts,avg_attn2_pts, max_attn2_pts], dim=1)

            # agg_pts = torch.cat([avg_attn_pts, max_attn_pts], dim=1)
            # B 9 W H
            # sig_cls = self.conv_squeeze_cls(agg_cls).sigmoid()
            # sig_pts = self.conv_squeeze_pts(agg_pts).sigmoid()
            alter_date = 803
            if alter_date == 803:
                r = self.conv_squeeze_mix(torch.cat([agg_cls,agg_pts], dim=1)).sogmoid()
            else:
                r = self.conv_squeeze_mix(torch.cat([agg_cls,agg_pts], dim=1))
            sig_cls = r[:,0:6]
            sig_pts = r[:,6:12]

            attn_mixd_cls0 = attn0_cls * sig_cls[:,0,:,:].unsqueeze(1) + attn1_cls * sig_cls[:,1,:,:].unsqueeze(1) + attn2_cls * sig_cls[:,2,:,:].unsqueeze(1)
            attn_mixd_cls1 = attn0_cls * sig_cls[:,3,:,:].unsqueeze(1) + attn1_cls * sig_cls[:,4,:,:].unsqueeze(1) + attn2_cls * sig_cls[:,5,:,:].unsqueeze(1)
            # attn_mixd_cls2 = attn0_cls * sig_cls[:,6,:,:].unsqueeze(1) + attn1_cls * sig_cls[:,7,:,:].unsqueeze(1) + attn2_cls * sig_cls[:,8,:,:].unsqueeze(1)

            attn_mixd_pts0 = attn0_pts * sig_pts[:,0,:,:].unsqueeze(1) + attn1_pts * sig_pts[:,1,:,:].unsqueeze(1) + attn2_pts * sig_pts[:,2,:,:].unsqueeze(1)
            attn_mixd_pts1 = attn0_pts * sig_pts[:,3,:,:].unsqueeze(1) + attn1_pts * sig_pts[:,4,:,:].unsqueeze(1) + attn2_pts * sig_pts[:,5,:,:].unsqueeze(1)
            # attn_mixd_pts2 = attn0_pts * sig_pts[:,6,:,:].unsqueeze(1) + attn1_pts * sig_pts[:,7,:,:].unsqueeze(1) + attn2_pts * sig_pts[:,8,:,:].unsqueeze(1)
            lsk_cls_res0 = self.lsk_conv_cmix_cls0(attn_mixd_cls0)  * cls_feat
            lsk_cls_res1 = self.lsk_conv_cmix_cls1(attn_mixd_cls1)  * cls_feat
            # lsk_cls_res2 = self.lsk_conv_cmix_cls(attn_mixd_cls2)


            lsk_pts_res0 = self.lsk_conv_cmix_pts0(attn_mixd_pts0)  * pts_feat
            lsk_pts_res1 = self.lsk_conv_cmix_pts1(attn_mixd_pts1)  * pts_feat
            # lsk_pts_res2 = self.lsk_conv_cmix_pts(attn_mixd_pts2)

            pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
            dcn_offset = pts_out_init_grad_mul - dcn_base_offset

            pts_x_mean = pts_out_init[:, 0::2].clone()#.mean(dim=1 )#.unsqueeze(1)
            pts_y_mean = pts_out_init[:, 1::2].clone()#.mean(dim=1)#.unsqueeze(1)
            pts_x_mean = torch.mean(pts_x_mean, dim=1 , keepdim=True)
            pts_y_mean = torch.mean(pts_y_mean, dim=1 , keepdim=True)
            core_pts = torch.cat([pts_x_mean, pts_y_mean], dim=1)
            core_pts = core_pts.repeat(1, 9, 1, 1)
            pts_grad_temp = (1 - self.gradient_mul) * core_pts.detach() + self.gradient_mul * core_pts
            core_offset = pts_grad_temp - dcn_base_offset
            # pts_div = self.div_reppoints_point(
            #     self.relu(self.div_reppoints_conv(pts_feat)))
            # pts_div_grad_mul = (1 - self.gradient_mul) * pts_div.detach() + self.gradient_mul * pts_out_init
            # pts_div_offset = pts_div_grad_mul - dcn_base_offset
            if False:
                # if True:

                plt.imshow(cls_feat[0, 0, :, :].cpu().detach().numpy())
                plt.title("cls_feat")
                plt.show()

            
            # cls_feat = self.my_lsk_cls(cls_feat)
            # pts_feat = self.my_lsk_pts(pts_feat)
            # -----end test--------
            if alter_date == 803:
                dcn_cls_feat = self.reppoints_cls_conv(lsk_cls_res0, dcn_offset) + self.l_conv1(lsk_cls_res1)  #+ self.pseudo_dcn_cls(lsk_cls_res1, core_offset) 
                dcn_pts_feat = self.reppoints_pts_refine_conv(lsk_pts_res0, dcn_offset) + self.l_conv2(lsk_pts_res1)  #+ self.pseudo_dcn_pts(lsk_pts_res1, core_offset)
            else:
                dcn_cls_feat = self.reppoints_cls_conv(lsk_cls_res0, dcn_offset) + self.pseudo_dcn_cls(lsk_cls_res1, core_offset) 
                dcn_pts_feat = self.reppoints_pts_refine_conv(lsk_pts_res0, dcn_offset) + self.pseudo_dcn_pts(lsk_pts_res1, core_offset)

            # 然后继续卷积,目标是分类
            cls_out = self.reppoints_cls_out(self.relu(dcn_cls_feat))
            # 以及,对代表点位置进行进一步微调,reppoints_pts_refine_conv是可形变卷积
            pts_out_refine = self.reppoints_pts_refine_out(self.relu(dcn_pts_feat))
            # 微调的结果加上基础值
            pts_out_refine = pts_out_refine + pts_out_init.detach()
            return cls_out, pts_out_init, pts_out_refine, x
        
        elif self.my_pts_mode == "sup_dcn":
            c1 = self.sup_dcn_conv(cls_feat)
            p16 = self.sup_dcn_out(self.relu(c1))
            p25 = torch.cat([pts_out_init, p16], dim= 1)

            p25_grad_mul = (1 - self.gradient_mul) * p25.detach() + self.gradient_mul * p25
            p25_dcn_offset = p25_grad_mul - self.sup_dcn_base_offset.type_as(x)
            


            dcn_cls_feat = self.sup_dcn(cls_feat, p25_dcn_offset)
            # 然后继续卷积,目标是分类
            cls_out = self.reppoints_cls_out(self.relu(dcn_cls_feat))

            pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
            dcn_offset = pts_out_init_grad_mul - dcn_base_offset
            # 以及,对代表点位置进行进一步微调,reppoints_pts_refine_conv是可形变卷积
            pts_out_refine = self.reppoints_pts_refine_out(
                self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
            # 微调的结果加上基础值
            pts_out_refine = pts_out_refine + pts_out_init.detach()
            return cls_out, pts_out_init, pts_out_refine, x
        elif self.my_pts_mode =="attn": # "attn":
            pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
            dcn_offset = pts_out_init_grad_mul - dcn_base_offset
            dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset)
            # 以及,对代表点位置进行进一步微调,reppoints_pts_refine_conv是可形变卷积
            dcn_loc_feat = self.reppoints_pts_refine_conv(pts_feat, dcn_offset)

            # TODO test modified----------
            B, Cannel, W, H = dcn_cls_feat.shape
            self.cls_strans.H, self.cls_strans.W = H, W
            self.loc_strans.H, self.loc_strans.W = H, W
            attn_mask = None
            # position embeding
            points = self.point_generators[0].grid_points(
                (W,H), 1).view(W,H,3)[:,:,:2]
            position_embeding_offset = pts_out_init[:,16:18].permute(0,2,3,1)
            position = torch.zeros_like( points.unsqueeze(0))  # position_embeding_offset
            position = position_embeding_offset + position_embeding_offset
            # position = self.norm(position)
            pe = torch.zeros(B,W,H, Cannel).float()
            pe.require_grad = False
            # div_term = (torch.arange(0, Cannel, 4).float() * -(math.log(10000.0) / Cannel)).exp().to('cuda') 

            # pe[:,:,:, 0::4] = torch.sin(position[:,:,:,0:1].repeat(1,1,1,64).to('cuda')  * div_term)
            # pe[:,:,:, 1::4] = torch.cos(position[:,:,:,0:1].repeat(1,1,1,64).to('cuda')   * div_term)
            # pe[:,:,:, 2::4] = torch.sin(position[:,:,:,1:2].repeat(1,1,1,64).to('cuda')   * div_term)
            # pe[:,:,:, 3::4] = torch.cos(position[:,:,:,1:2].repeat(1,1,1,64).to('cuda')   * div_term)

            pe[:,:,:, 0] = torch.sin(position[:,:,:,0])
            pe[:,:,:, 1] = torch.cos(position[:,:,:,0])
            pe[:,:,:, 2] = torch.sin(position[:,:,:,1])
            pe[:,:,:, 3] = torch.cos(position[:,:,:,1])
            # embeding_token = self.ape(position)
            # embeding_token = self.relu(embeding_token)
            embeding_token = pe.to('cuda')
            dcn_cls_feat_embeded =  dcn_cls_feat.permute(0,2,3,1) + embeding_token
            dcn_loc_feat_embeded =  dcn_loc_feat.permute(0,2,3,1) + embeding_token
            # end position embeding
            dcn_cls_feat_trans_input = dcn_cls_feat_embeded.view(B, H*W, Cannel)
            dcn_loc_feat_trans_input = dcn_loc_feat_embeded.view(B, H*W, Cannel)
            dcn_cls_feat_res = self.cls_strans(dcn_cls_feat_trans_input, attn_mask)
            dcn_loc_feat_res = self.loc_strans(dcn_loc_feat_trans_input, attn_mask)
            dcn_cls_feat_trans_output = dcn_cls_feat_res.permute(0,2,1).view(B, Cannel, W, H)
            dcn_loc_feat_trans_output = dcn_loc_feat_res.permute(0,2,1).view(B, Cannel, W, H)
            # dcn_cls_feat_attn = self.position_attn(dcn_cls_feat,pts_out_init,4)
            # dcn_loc_feat_attn = self.position_attn(dcn_loc_feat,pts_out_init,4)
            cls_out = self.reppoints_cls_out(self.relu(dcn_cls_feat_trans_output))
            pts_out_refine = self.reppoints_pts_refine_out(self.relu(dcn_loc_feat_trans_output))
            # -----end test--------
            # 然后继续卷积,目标是分类
            # cls_out = self.reppoints_cls_out(self.relu(dcn_cls_feat))
            # pts_out_refine = self.reppoints_pts_refine_out(self.relu(dcn_loc_feat))
            # 微调的结果加上基础值
            pts_out_refine = pts_out_refine + pts_out_init.detach()

            return cls_out, pts_out_init, pts_out_refine, x

            # cls_out = self.reppoints_cls_out(self.relu(dcn_cls_feat))
            # pts_out_refine = self.reppoints_pts_refine_out(self.relu(dcn_loc_feat))
            # 分离横坐标和纵坐标
            # y_pts_shift = yx_pts_shift[..., 0::2]
            # x_pts_shift = yx_pts_shift[..., 1::2]
            # xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
            # xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
            # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            # attn = (q @ k.transpose(-2, -1))
            # attn = self.softmax(attn)
            # attn = self.attn_drop(attn)
            # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            # x = self.proj(x)
            # x = self.proj_drop(x)

        # end
        # 细化并且对代表点分类
        # gradient_mul 反向传播梯度的因子,作用是控制反向传播时的比例
        # pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
        # dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        # copy
        if self.my_pts_mode == "pts_down" or self.my_pts_mode == "pts_up":
            pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
            dcn_offset = pts_out_init_grad_mul - dcn_base_offset
            pts_div_grad_mul = (1 - 0.3) * pts_div.detach() +  0.3* pts_div 
            # pts_div_grad_mul = pts_div[:,0:2].repeat(1,9,1,1) # (1 - self.gradient_mul) * pts_div.detach() +  self.gradient_mul * 
            pts_div_offset = pts_div_grad_mul #- dcn_base_offset 
            # TODO 你不是全为零吗,满足你
            # pts_div = dcn_offset.new_zeros(dcn_offset.shape) #+ dcn_base_offset
            # end

            # pts_div_offset = pts_div - dcn_base_offset
        
            
        # end

        # 对代表点分类,先用可形变卷积提取代表点点特征

        # dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset)
        # rewrite 1line
        if self.my_pts_mode == "pts_down" :
            dcn_cls_feat = self.reppoints_cls_conv(cls_feat, pts_div_offset)
            cls_out = self.reppoints_cls_out(self.relu(dcn_cls_feat))

        elif self.my_pts_mode == "pts_up":
            dcn_cls_feat = self.reppoints_cls_conv(cls_feat, pts_div_offset)
            # cls_out = self.reppoints_cls_out(self.relu(dcn_cls_feat))
            # dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset)
            # 然后继续卷积,目标是分类
            cls_out = self.reppoints_cls_out(self.relu(dcn_cls_feat))
            # 以及,对代表点位置进行进一步微调,reppoints_pts_refine_conv是可形变卷积
            pts_out_refine = self.reppoints_pts_refine_out(
                self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
            # 微调的结果加上基础值
            pts_out_refine = pts_out_refine + pts_out_init.detach()

            pts_div_offset_golbal = pts_div_offset + dcn_base_offset

            # return cls_out, pts_div_offset_golbal, pts_out_refine, x
            return cls_out, pts_out_init, pts_out_refine, x
        
        # if self.my_pts_mode == "com1" or self.my_pts_mode == "com3":
        elif self.my_pts_mode == "core":
            alter = True
            if alter == True:
                pts_x_mean = pts_out_init[:, 0::2].mean(dim=1).unsqueeze(1)
                pts_y_mean = pts_out_init[:, 1::2].mean(dim=1).unsqueeze(1)
                core_pts = torch.cat([pts_x_mean, pts_y_mean], dim=1).repeat(1, 9, 1, 1)
                pts_grad_temp = (1 - self.gradient_mul) * core_pts.detach() + self.gradient_mul * core_pts
                core_offset = pts_grad_temp - dcn_base_offset
            else:
                offset_x_mean = dcn_offset[:, 0::2].mean(dim=1).unsqueeze(1)
                offset_y_mean = dcn_offset[:, 1::2].mean(dim=1).unsqueeze(1)
                core_offset = torch.cat([offset_x_mean, offset_y_mean], dim=1).repeat(1, 9, 1, 1)
            # TODO 是否
            # core_offset = core_offset.detach()
            # 伪-单点可形变卷积
            dcn_cls_feat = self.reppoints_cls_conv(cls_feat, core_offset)
            cls_out = self.reppoints_cls_out(dcn_cls_feat)
            dcn_temp = self.reppoints_pts_refine_conv(pts_feat, dcn_offset)
            # pts_refine_temp = dcn_temp + pts_feat  # + self.conv3(pts_feat)+ self.conv1(pts_feat) 
            pts_out_refine = self.reppoints_pts_refine_out(self.relu(dcn_temp))
            pts_out_refine = pts_out_refine + pts_out_init.detach()
        
        elif self.my_pts_mode == "core_v4":
            pts_x_mean = pts_out_init[:, 0::2].mean(dim=1).unsqueeze(1)
            pts_y_mean = pts_out_init[:, 1::2].mean(dim=1).unsqueeze(1)
            core_pts = torch.cat([pts_x_mean, pts_y_mean], dim=1).repeat(1, 9, 1, 1)
            pts_grad_temp = (1 - self.gradient_mul) * core_pts.detach() + self.gradient_mul * core_pts
            core_offset = pts_grad_temp - dcn_base_offset
            # TODO 是否
            # core_offset = core_offset.detach()
            # 伪-单点可形变卷积
            dcn_cls_feat = self.reppoints_cls_conv(cls_feat, core_offset)
            cls_out = self.reppoints_cls_out(dcn_cls_feat)
            dcn_temp = self.reppoints_pts_refine_conv(pts_feat, core_offset)
            # pts_refine_temp = dcn_temp + pts_feat  # + self.conv3(pts_feat)+ self.conv1(pts_feat) 
            pts_out_refine = self.reppoints_pts_refine_out(self.relu(dcn_temp))
            pts_out_refine = pts_out_refine + pts_out_init.detach()

        elif self.my_pts_mode == "core_v2":

            # 改正一个错误, core应该对pts进行而不是offset
            # 因为dcn_base_offset有偏移， 否则是中心开花而不是单点
            alter = True  #_v2_alter/epoch_2_0715_1e-6.pth' /1  0.9246 /9 0.9248
            # alter = False
            if alter == True:
                pts_x_mean = pts_out_init[:, 0::2].mean(dim=1).unsqueeze(1)
                pts_y_mean = pts_out_init[:, 1::2].mean(dim=1).unsqueeze(1)
                single_core_pts = torch.cat([pts_x_mean, pts_y_mean], dim=1)
                core_pts = single_core_pts.repeat(1, 9, 1, 1)
                core_pts_grad_temp = (1 - self.gradient_mul) * core_pts.detach() + self.gradient_mul * core_pts
                core_offset = core_pts_grad_temp - dcn_base_offset
                ct_pts_vec = (pts_out_init - core_pts).detach()

                # core_offset =  torch.zeros_like(core_pts_grad_temp)- dcn_base_offset
                # TODO 是否drop
                drop_inds = torch.randint(0, pts_out_init.shape[1]//2,[1])
                pts_out_init = pts_out_init.clone()
                pts_out_init[:,drop_inds*2:drop_inds*2+2,:,:] = single_core_pts.detach()
                pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
                dcn_offset = pts_out_init_grad_mul - dcn_base_offset
            else:
                offset_x_mean = dcn_offset[:, 0::2].mean(dim=1).unsqueeze(1)
                offset_y_mean = dcn_offset[:, 1::2].mean(dim=1).unsqueeze(1)
                core_offset = torch.cat([offset_x_mean, offset_y_mean], dim=1).repeat(1, 9, 1, 1)

            # TODO 是否
            # core_offset = core_offset.detach()
            # 伪-单点可形变卷积
            # dcn_cls_feat = self.reppoints_cls_conv(cls_feat, core_offset)
            # 漏了个激活函数我的
            # cls_out = self.reppoints_cls_out(dcn_cls_feat)
            # div_term = torch.tensor(9).type_as(x)
            # 对照实验
            after_date = 729
            # after_date = 730
            if after_date == 713:
                # 附带余弦相似度的修改line917
                dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset) + self.pseudo_dcn_cls(cls_feat, core_offset)#  / torch.tensor(9).type_as(x)
            elif after_date == 729:
                dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset) 
                dcn_cls_feat2 =  self.pseudo_dcn_cls(cls_feat, core_offset)#  / torch.tensor(9).type_as(x)
                dcn_pts_feat = self.reppoints_pts_refine_conv(pts_feat, dcn_offset)
                dcn_pts_feat2 = self.pseudo_dcn_pts(pts_feat, core_offset)
                # 又忘了激活函数
                # mix1 = self.ddim_conv1(torch.cat([dcn_cls_feat,dcn_cls_feat2],dim=1))
                # mix2 = self.ddim_conv2(torch.cat([dcn_pts_feat,dcn_pts_feat2],dim=1))
                mix1 = self.ddim_conv1(self.relu(torch.cat([dcn_cls_feat,dcn_cls_feat2],dim=1)))
                mix2 = self.ddim_conv2(self.relu(torch.cat([dcn_pts_feat,dcn_pts_feat2],dim=1)))

                cls_out = self.reppoints_cls_out(self.relu(mix1))
                pts_out_refine = self.reppoints_pts_refine_out(self.relu(mix2))
                pts_out_refine = pts_out_refine + pts_out_init.detach()
                return cls_out, pts_out_init, pts_out_refine, x
            elif after_date == 730:
                dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset) 
                dcn_cls_feat2 =  self.pseudo_dcn_cls(cls_feat, core_offset)#  / torch.tensor(9).type_as(x)
                dcn_pts_feat = self.reppoints_pts_refine_conv(pts_feat, dcn_offset)
                dcn_pts_feat2 = self.pseudo_dcn_pts(pts_feat, core_offset)
                # 又忘了激活函数
                # mix1 = self.ddim_conv1(torch.cat([dcn_cls_feat,dcn_cls_feat2],dim=1))
                # mix2 = self.ddim_conv2(torch.cat([dcn_pts_feat,dcn_pts_feat2],dim=1))
                mix1 = self.ddim_conv1(self.relu(torch.cat([dcn_cls_feat,dcn_cls_feat2],dim=1)))
                mix2 = self.ddim_conv2(self.relu(torch.cat([dcn_pts_feat,dcn_pts_feat2],dim=1)))
                ct_rate = self.ct_rate_conv1(self.relu(mix1))
                ct_pts_vec = torch.ones_like(pts_out_init) * dcn_base_offset
                ct_pts_grad_temp = (ct_pts_vec*(0+ct_rate.repeat_interleave(2,1)) + core_pts).detach() # (1 - self.gradient_mul) * ct_pts.detach() + self.gradient_mul * ct_pts
                ct_offset = ct_pts_grad_temp - dcn_base_offset
                ct_cls_out = self.relu(self.ct_dcn_cls(x.detach(),ct_offset))
                ct_pts_out = self.relu(self.ct_dcn_pts(x.detach(),ct_offset))
                mix1 = mix1 + ct_cls_out
                mix2 = mix2 + ct_pts_out
                cls_out = self.reppoints_cls_out(self.relu(mix1))
                pts_out_refine = self.reppoints_pts_refine_out(self.relu(mix2))
                pts_out_refine = pts_out_refine + pts_out_init.detach()
                # return cls_out, ct_pts_grad_temp, pts_out_refine, x
                return cls_out, pts_out_init, pts_out_refine, x
            
            elif after_date == 712:
                dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset)   # v2/epoch_9_10e-6_0.9286.pth  0.9112
            # elif after_date == 711:
            #     dcn_cls_feat = self.pseudo_dcn_cls(cls_feat, core_offset)
            else:
                dcn_cls_feat = self.reppoints_cls_conv(cls_feat, core_offset)  # v2/epoch_9_10e-6_0.9286.pth  0.8089
            
                # dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset)

                cls_out = self.reppoints_cls_out(self.relu(dcn_cls_feat))
                
                dcn_temp = self.reppoints_pts_refine_conv(pts_feat, dcn_offset)
                # 伪-单点可形变卷积
                pts_refine_temp = dcn_temp  + self.pseudo_dcn_pts(pts_feat, core_offset) #/ torch.tensor(9).type_as(x)#  + self.conv1(pts_feat)  #/ torch.tensor(9).type_as(x)# pts_feat  # + self.conv3(pts_feat)
                pts_out_refine = self.reppoints_pts_refine_out(self.relu(pts_refine_temp))
                pts_out_refine = pts_out_refine + pts_out_init.detach()
            return cls_out, pts_out_init, pts_out_refine, x
            
        elif self.my_pts_mode == "core_v3":

            # 因为dcn_base_offset有偏移， 否则是中心开花而不是单点 even False better
            alter = False
            if alter == True:
                # pts_x_mean = pts_out_init[:, 0::2].mean(dim=1).unsqueeze(1)
                # pts_y_mean = pts_out_init[:, 1::2].mean(dim=1).unsqueeze(1)
                pts_x_mean = pts_out_init[:, 0::2].mean(dim=1, keepdim=True)
                pts_y_mean = pts_out_init[:, 1::2].mean(dim=1, keepdim=True)
                core_pts = torch.cat([pts_x_mean, pts_y_mean], dim=1).repeat(1, 9, 1, 1)
                pts_grad_temp = (1 - self.gradient_mul) * core_pts.detach() + self.gradient_mul * core_pts
                core_offset = pts_grad_temp - dcn_base_offset
            else:
                # offset_x_mean = dcn_offset[:, 0::2].mean(dim=1).unsqueeze(1)
                # offset_y_mean = dcn_offset[:, 1::2].mean(dim=1).unsqueeze(1)
                offset_x_mean = dcn_offset[:, 0::2].mean(dim=1, keepdim=True)
                offset_y_mean = dcn_offset[:, 1::2].mean(dim=1, keepdim=True)
                core_offset = torch.cat([offset_x_mean, offset_y_mean], dim=1).repeat(1, 9, 1, 1)
        
            dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset) + self.pseudo_dcn_cls(cls_feat, core_offset) / torch.tensor(9).type_as(x) #+ cls_feat

            cls_out = self.reppoints_cls_out(self.relu(dcn_cls_feat))
            pts_refine_temp = self.reppoints_pts_refine_conv(pts_feat, dcn_offset) + self.pseudo_dcn_pts(pts_feat, core_offset) / torch.tensor(9).type_as(x)
            pts_out_refine = self.reppoints_pts_refine_out(self.relu(pts_refine_temp))
            pts_out_refine = pts_out_refine + pts_out_init.detach()
        elif self.my_pts_mode == "ide2": #不变定位, 只动分类
            pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
            dcn_offset = pts_out_init_grad_mul - dcn_base_offset
            sel = self.sele_1(x)
            sel = self.sele_2(self.relu(sel))

            # pts_x_mean = pts_out_init[:, 0::2].mean(dim=1).unsqueeze(1)
            # pts_y_mean = pts_out_init[:, 1::2].mean(dim=1).unsqueeze(1)
            # core_pts = torch.cat([pts_x_mean, pts_y_mean], dim=1).repeat(1, 9, 1, 1)
            # pts_grad_temp = (1 - self.gradient_mul) * core_pts.detach() + self.gradient_mul * core_pts
            # core_offset = pts_grad_temp - dcn_base_offset
            # sel_soft = torch.nn.functional.softmax(sel, dim=1)
            sel_soft = sel.sigmoid()
            conv_res = self.conv1(cls_feat)
            dcn_res = self.reppoints_cls_conv(cls_feat, dcn_offset)
            soft_res = conv_res*sel_soft[:,0:1] + dcn_res*sel_soft[:,1:2] #+ core_res*sel_soft[:,2:3]
            # soft_res = conv_res
            # cls_out = self.reppoints_cls_out(self.relu_no_inplace(soft_res))
            cls_out = self.reppoints_cls_out(self.relu(soft_res))
            # cls_out = self.reppoints_cls_out(cls_feat)
            # cls_out = self.reppoints_cls_out(self.conv1(cls_feat* torch.tensor(self.num_points).type_as(x)))
            #    plt.imshow(cls_feat[0, 0, :, :].cpu().detach().numpy())
            #     plt.title("cls_feat")
            #     plt.show()
            # cls_dcn_res= self.pseudo_dcn_cls(cls_feat.detach(), dcn_offset.detach())
            # cls_ass_res= self.conv3(pts_feat)
            # cls_out_refine = self.cls_pts_assis_refine_out(self.relu(cls_ass_res))
            # cls_out_refine = cls_out_refine[:,0:2].repeat(1,9,1,1)
            # dcn_temp = self.reppoints_pts_refine_conv(pts_feat, dcn_offset)* sel_soft[:,2:3] + cls_ass_res * sel_soft[:,3:4]
            dcn_temp = self.reppoints_pts_refine_conv(pts_feat, dcn_offset)
            # dcn_temp = self.conv_cat(torch.cat([dcn_temp, soft_res], dim=1))

            # norefine_temp = dcn_temp + self.conv1(pts_feat) + pts_feat  # + self.conv3(pts_feat)
            pts_out_refine_offset = self.reppoints_pts_refine_out(self.relu(dcn_temp))
            pts_out_refine = pts_out_refine_offset + pts_out_init.detach()

            two_path_out_refine = pts_out_refine_offset.detach() + pts_out_init.detach()
            # forward_dcn_temp= self.forward_pts_refine_conv(pts_feat, (two_path_out_refine - dcn_base_offset))
            # forward_pts_out_refine_offset = self.forward_pts_refine_out(self.relu(forward_dcn_temp))
            forward_dcn_temp= self.reppoints_pts_refine_conv(pts_feat.detach(), (two_path_out_refine - dcn_base_offset))
            forward_pts_out_refine_offset = self.reppoints_pts_refine_out(self.relu(forward_dcn_temp))
            forward_pts_out_refine = two_path_out_refine.detach() + forward_pts_out_refine_offset

            pts_out_refine_avg = pts_out_refine *(1- sel_soft[:,2:3] ) + forward_pts_out_refine * sel_soft[:,2:3]
            # pts_out_refine = pts_out_refine + cls_out_refine #* sel_soft[:,2:3]
            return cls_out, pts_out_init, pts_out_refine_avg, x
            return cls_out, pts_out_init, pts_out_refine, x

        elif self.my_pts_mode == "ide3": #不变定位, 只动分类
            
            cls_temp = cls_feat+self.relu(self.conv1(cls_feat)) + self.relu(self.conv_ide3(cls_feat))
            cls_out = self.reppoints_cls_out(cls_temp)

            dcn_temp = self.reppoints_pts_refine_conv(pts_feat, dcn_offset)
            # norefine_temp = dcn_temp + self.conv1(pts_feat) + pts_feat  # + self.conv3(pts_feat)
            pts_out_refine = self.reppoints_pts_refine_out(self.relu(dcn_temp))
            pts_out_refine = pts_out_refine + pts_out_init.detach()

        elif self.my_pts_mode == "int":
            dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset)
            cls_temp = cls_feat+dcn_cls_feat+self.div_common_conv1(cls_feat)
            cls_out = self.reppoints_cls_out(self.relu(cls_temp))
            dcn_temp = self.reppoints_pts_refine_conv(pts_feat, dcn_offset)
            norefine_temp = dcn_temp + self.conv1(pts_feat) + pts_feat  # + self.conv3(pts_feat)
            pts_out_refine = self.reppoints_pts_refine_out(self.relu(norefine_temp))
            pts_out_refine = pts_out_refine + pts_out_init.detach()

        elif self.my_pts_mode == "com1":
             # cls_out = self.reppoints_cls_out(self.relu_no_inplace(cls_feat))
            cls_out = self.reppoints_cls_out(self.div_common_conv1(cls_feat))
            dcn_temp = self.reppoints_pts_refine_conv(pts_feat, dcn_offset)
            norefine_temp = dcn_temp + self.conv1(pts_feat) + pts_feat  # + self.conv3(pts_feat)
            pts_out_refine = self.reppoints_pts_refine_out(self.relu(norefine_temp))
            pts_out_refine = pts_out_refine + pts_out_init.detach()

        elif self.my_pts_mode == "ide":
            # cls_out = self.reppoints_cls_out(self.relu_no_inplace(cls_feat))
            cls_out = self.reppoints_cls_out(cls_feat)

            dcn_temp = self.reppoints_pts_refine_conv(pts_feat, dcn_offset)
            norefine_temp = dcn_temp + self.conv1(pts_feat) + pts_feat  # + self.conv3(pts_feat)
            pts_out_refine = self.reppoints_pts_refine_out(self.relu(norefine_temp))

            pts_out_refine = pts_out_refine + pts_out_init.detach()
        
        elif self.my_pts_mode == "mean":
             # TODO PTS EXTRA
        
            B, Cannel, W, H = x.shape
            
            grid_w = torch.arange(0,W).view(1,1,W,1).repeat(1,1,1,H)
            grid_h = torch.arange(0,H).view(1,1,1,H).repeat(1,1,W,1)
            # TODO  先后顺序？
          
            grid = torch.cat([grid_w,grid_h], dim=1) #.repeat(B,self.num_points,1,1).type_as(x)
            grid = grid.permute(0,2,3,1)
            # points = self.point_generators[0].grid_points((W,H), 1).view(W,H,3)[:,:,:2].unsqueeze(0)
            # position = points + pts_out_init
            # h = x.shape[2]
            # w = x.shape[3]
            locations1 = grid.view(grid.shape[0], grid.shape[1] * grid.shape[2], 1, 2)
            # locations1 = points.view(points.shape[0], points.shape[1] * points.shape[2], 1, 2)
            locations2 = locations1.repeat(1,1,self.num_points,1).to('cuda').type_as(cls_feat)
            locations = locations2 + pts_out_init.permute(0,2,3,1).view(grid.shape[0], grid.shape[1] * grid.shape[2], self.num_points, 2)
            # locations = locations.view(locations.shape[0], locations.shape[1] * locations.shape[2], -1, 2).clone()
            # # 先映射到[0,1],再映射到[-1,1]区间上
            locations[..., 0] = locations[..., 0] / ((W-1) / 2.) - 1
            locations[..., 1] = locations[..., 1] / ((H-1) / 2.) - 1

            feature3 = nn.functional.grid_sample(cls_feat, locations.flip(-1), align_corners=True)#, mode='nearest')
            feature_2d = feature3.mean(-1).view(B, Cannel, W, H) 

            feature3_pts = nn.functional.grid_sample(pts_feat, locations.flip(-1), align_corners=True)#, mode='nearest')
            feature_2d_pts = feature3_pts.mean(-1).view(B, Cannel, W, H) 
            # dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset)
            # 然后继续卷积,目标是分类
            cls_out = self.reppoints_cls_out(feature_2d)
            # 以及,对代表点位置进行进一步微调,reppoints_pts_refine_conv是可形变卷积
            pts_out_refine = self.reppoints_pts_refine_out(feature_2d_pts)
            # 微调的结果加上基础值
            pts_out_refine = pts_out_refine + pts_out_init.detach()
            return cls_out, pts_out_init, pts_out_refine, x
        
        elif self.my_pts_mode == "demo":
            # TODO test modified
            # pts_x_mean = pts_out_init[:, 0::2].clone()#.mean(dim=1 )#.unsqueeze(1)
            # pts_y_mean = pts_out_init[:, 1::2].clone()#.mean(dim=1)#.unsqueeze(1)
            # pts_x_mean = torch.mean(pts_x_mean, dim=1 , keepdim=True)
            # pts_y_mean = torch.mean(pts_y_mean, dim=1 , keepdim=True)
            # core_pts = torch.cat([pts_x_mean, pts_y_mean], dim=1)
            # #.repeat(1, 9, 1, 1)
            # drop_inds = torch.randint(0, pts_out_init.shape[1]//2,[1])
            # pts_out_init[:,drop_inds*2:drop_inds*2+2,:,:] = core_pts

            # B, Cannel, W, H = x.shape
            
            # grid_w = torch.arange(0,W).view(1,1,W,1).repeat(1,1,1,H)
            # grid_h = torch.arange(0,H).view(1,1,1,H).repeat(1,1,W,1)
            # # TODO  先后顺序？
          
            # grid = torch.cat([grid_w,grid_h], dim=1) #.repeat(B,self.num_points,1,1).type_as(x)
            # grid = grid.permute(0,2,3,1)
            # # points = self.point_generators[0].grid_points((W,H), 1).view(W,H,3)[:,:,:2].unsqueeze(0)
            # # position = points + pts_out_init
            # # h = x.shape[2]
            # # w = x.shape[3]
            # num_points = 1
            # locations1 = grid.view(grid.shape[0], grid.shape[1] * grid.shape[2], 1, 2)
            # # locations1 = points.view(points.shape[0], points.shape[1] * points.shape[2], 1, 2)
            # locations2 = locations1.repeat(1,1,num_points,1).to('cuda').type_as(cls_feat)
            # locations = locations2 + core_pts.permute(0,2,3,1).view(grid.shape[0], grid.shape[1] * grid.shape[2], num_points, 2)
            # # locations = locations.view(locations.shape[0], locations.shape[1] * locations.shape[2], -1, 2).clone()
            # # # 先映射到[0,1],再映射到[-1,1]区间上
            # locations[..., 0] = locations[..., 0] / ((W-1) / 2.) - 1
            # locations[..., 1] = locations[..., 1] / ((H-1) / 2.) - 1

            # feature3 = nn.functional.grid_sample(cls_feat, locations.flip(-1), align_corners=True)#, mode='nearest')

            # feature3_pts = nn.functional.grid_sample(pts_feat, locations.flip(-1), align_corners=True)#, mode='nearest')
            # # B C H*W 9
            # # mean
            # feature_2d = feature3.mean(-1).view(B, Cannel, W, H) 
            # feature_2d_pts = feature3_pts.mean(-1).view(B, Cannel, W, H) 
            # # TODO add attn

            # # ft1 = feature3.permute(0,2,3,1).view(-1,9,Cannel)            
            # # B_, N, C = ft1.shape
            # # num_heads = 1
            # # qkv1 = self.qkv(ft1)
            # # qkv = qkv1.reshape(B_, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
            # # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            # # qkv_scale =  (C // num_heads) ** -0.5
            # # q = q * qkv_scale
            # # attn = (q @ k.transpose(-2, -1))

            # # attn = self.attn_drop(attn)

            # # res = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            # # res = ft1
            # # res = self.proj(res)
            # # res = self.proj_drop(res)
            # # attn_ft = res.mean(1).view(B, Cannel, W, H) 
            # # temp = ft1.reshape(B, Cannel, W* H) 
            # #feature3 ==ft1.view(B,W*H,9,Cannel).permute(0,3,1,2)
            # # attn_ft = ft1.mean(1).reshape(B, Cannel, W, H) 
            # # attn_ft = res.view(B,W*H,9,Cannel).reshape(B, W, H,9,Cannel).mean(-2).permute(0,3,1,2)
            # # cls_out = self.reppoints_cls_out(attn_ft)
            # cls_out = self.reppoints_cls_out(feature_2d)
            # # 以及,对代表点位置进行进一步微调,reppoints_pts_refine_conv是可形变卷积
            # pts_out_refine = self.reppoints_pts_refine_out(feature_2d_pts)
            # # 微调的结果加上基础值
            # pts_out_refine = pts_out_refine + pts_out_init.detach()
            # return cls_out, pts_out_init, pts_out_refine, x
            #----------------------------------------
            # B, Cannel, W, H = x.shape
            
            # grid_w = torch.arange(0,W).view(1,1,W,1).repeat(1,1,1,H)
            # grid_h = torch.arange(0,H).view(1,1,1,H).repeat(1,1,W,1)
            # # TODO  先后顺序？
          
            # grid = torch.cat([grid_w,grid_h], dim=1) #.repeat(B,self.num_points,1,1).type_as(x)
            # grid = grid.permute(0,2,3,1)
            # locations1 = grid.view(grid.shape[0], grid.shape[1] * grid.shape[2], 1, 2)
            # locations2 = locations1.repeat(1,1,self.num_points,1).to('cuda').type_as(cls_feat)
            # locations = locations2 + pts_out_init.permute(0,2,3,1).view(grid.shape[0], grid.shape[1] * grid.shape[2], self.num_points, 2)
            # # # 先映射到[0,1],再映射到[-1,1]区间上
            # locations[..., 0] = locations[..., 0] / ((W-1) / 2.) - 1
            # locations[..., 1] = locations[..., 1] / ((H-1) / 2.) - 1

            # feature3 = nn.functional.grid_sample(cls_feat, locations.flip(-1), align_corners=True)#, mode='nearest')
            # feature_2d = feature3.mean(-1).view(B, Cannel, W, H) 

            # feature3_pts = nn.functional.grid_sample(pts_feat, locations.flip(-1), align_corners=True)#, mode='nearest')
            # feature_2d_pts = feature3_pts.mean(-1).view(B, Cannel, W, H) 
            # # dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset)
            # # 然后继续卷积,目标是分类
            # mix1 = self.ddim_conv1(torch.cat([cls_feat,feature_2d],dim=1))
            # mix2 = self.ddim_conv2(torch.cat([pts_feat,feature_2d_pts],dim=1))
            # cls_out = self.reppoints_cls_out(mix1)
            # # 以及,对代表点位置进行进一步微调,reppoints_pts_refine_conv是可形变卷积
            # pts_out_refine = self.reppoints_pts_refine_out(mix2)
            # # 微调的结果加上基础值
            # pts_out_refine = pts_out_refine + pts_out_init.detach()
            # return cls_out, pts_out_init, pts_out_refine, x
            # cos_similarity = nn.CosineSimilarity(dim=2, eps=1e-6)
            # weit = self.reppoints_cls_conv.state_dict()['weight'].view(256,256,9).permute(0,2,1)
            # sim = cos_similarity(weit[:,0:1],weit)
            # sm = sim.mean(dim=0)
            # ----------------------
            # lsk_c0 = cls_feat
            # lsk_c1 = self.lsk_conv_spatial0(lsk_c0)
            # lsk_c2 = self.lsk_conv_spatial1(lsk_c1)

            # attn0 = self.lsk_conv1(cls_feat)
            # attn1 = self.lsk_conv2(lsk_c1)
            # attn2 = self.lsk_conv3(lsk_c2)
            
            # # attn = torch.cat([attn1, attn2], dim=1)
            # avg_attn0 = torch.mean(attn0, dim=1, keepdim=True)
            # max_attn0, _ = torch.max(attn0, dim=1, keepdim=True)
            # avg_attn1 = torch.mean(attn1, dim=1, keepdim=True)
            # max_attn1, _ = torch.max(attn1, dim=1, keepdim=True)
            # avg_attn2 = torch.mean(attn2, dim=1, keepdim=True)
            # max_attn2, _ = torch.max(attn2, dim=1, keepdim=True)
            # agg = torch.cat([avg_attn0, max_attn0,avg_attn1, max_attn1,avg_attn2, max_attn2], dim=1)
            # sig = self.conv_squeeze(agg).sigmoid()
            # attn_mixd = attn0 * sig[:,0,:,:].unsqueeze(1) + attn1 * sig[:,1,:,:].unsqueeze(1) + attn2 * sig[:,2,:,:].unsqueeze(1)
            # lsk_res = self.lsk_conv_cmix(attn_mixd)
            # shift_inds = torch.randint(0, pts_out_init.shape[1]//2,[1])
            # c1 = torch.zeros_like(pts_out_init)
            # # l1 = pts_out_init[:,2:4]
            # # hel1 = torch.zeros_like(pts_out_init[:,2:4])
            # # hel1[:,0] = pts_out_init[:,3] * pts_out_init[:,4]
            # # hel1[:,1] = -pts_out_init[:,2] * pts_out_init[:,4]
            # c1[:,0:2] = pts_out_init[:,10:12] 
            # c1[:,2:4] = pts_out_init[:,12:14] 
            # c1[:,4:6] = pts_out_init[:,14:16] 
            # c1[:,6:8] = pts_out_init[:,16:18]
            # c1[:,8:10] = pts_out_init[:,8:10] #- hel1
            # # c1[:,10:12] = pts_out_init[:,0:2] + l1 + hel1
            # # c1[:,12:14] = pts_out_init[:,0:2] + hel1 -l1
            # # c1[:,14:16] = pts_out_init[:,0:2] - l1 -hel1
            # # c1[:,16:18] = pts_out_init[:,0:2] - hel1 + l1
            # c1[:,10:12] = pts_out_init[:,10:12] 
            # c1[:,12:14] = pts_out_init[:,12:14] 
            # c1[:,14:16] = pts_out_init[:,14:16] 
            # c1[:,16:18] = pts_out_init[:,16:18]
            # pts_out_init = c1
            # cls_feat = self.my_lsk_cls(cls_feat)
            # pts_out_init[:,0:16] = pts_out_init[:,16:18].repeat(1,8,1,1) + pts_out_init[:,0:16]
            # shift_pts = pts_out_init.clone()
            # shift_pts[:,0:2] = pts_out_init[:,4:6]
            # shift_pts[:,2:4] = pts_out_init[:,14:16]
            # shift_pts[:,4:6] = pts_out_init[:,6:8]
            # shift_pts[:,14:16] = pts_out_init[:,0:2]
            # shift_pts[:,6:8] = pts_out_init[:,2:4]
            # shift_dcn_offset = shift_pts  - dcn_base_offset
            # pts_out_init = shift_pts
            # -----------------
            pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
            dcn_offset = pts_out_init_grad_mul - dcn_base_offset

            # shift_pts = pts_out_init.clone()
            # shift_pts[:,0:2] = pts_out_init[:,2:4]
            # shift_pts[:,2:4] = pts_out_init[:,4:6]
            # shift_pts[:,4:6] = pts_out_init[:,14:16]
            # shift_pts[:,14:16] = pts_out_init[:,6:8]
            # shift_pts[:,6:8] = pts_out_init[:,0:2]
            # shift_dcn_offset = shift_pts  - dcn_base_offset

            pts_x = pts_out_init[:, 0::2].clone()#.mean(dim=1 )#.unsqueeze(1)
            pts_y = pts_out_init[:, 1::2].clone()#.mean(dim=1)#.unsqueeze(1)
            pts_x_mean = torch.mean(pts_x, dim=1 , keepdim=True)
            pts_y_mean = torch.mean(pts_y, dim=1 , keepdim=True)
            pts_x_max, _ = torch.max(pts_x, dim=1 , keepdim=True)
            pts_y_max, _ = torch.max(pts_y, dim=1 , keepdim=True)
            pts_x_min, _ = torch.min(pts_x, dim=1 , keepdim=True)
            pts_y_min, _ = torch.min(pts_y, dim=1 , keepdim=True)
            pts_feature_value = torch.cat([pts_x_mean, pts_y_mean, pts_x_max, pts_y_max, pts_x_min, pts_y_min], dim =1)
            core_pts = torch.cat([pts_x_mean, pts_y_mean], dim=1).repeat(1,9,1,1)

            if False:
                # if True:
                # zero_offset = dcn_offset.new_zeros(dcn_offset.shape) - dcn_base_offset
                # zero_dcn_cls_feat = self.reppoints_cls_conv(cls_feat, zero_offset)
                # import matplotlib.pyplot as plt
                # 
                plt.imshow(cls_feat[0, 0, :, :].cpu().detach().numpy())
                plt.title("cls_feat")
                plt.show()
                plt.imshow(pts_out_init_grad_mul[0, 0, :, :].cpu().detach().numpy())
                plt.title("pts_out_init_grad_mul")
                plt.show()
                plt.imshow(pts_feat[0, 0, :, :].cpu().detach().numpy())
                plt.title("pts_feat")
                plt.show()
            # sup_dcn_points = self.sup_dcn_out_points(pts_feat)
            # sup_dcn_points_1 = self.sup_dcn_out_points_head1(pts_feat)
            # sup_dcn_points_2 = self.sup_dcn_out_points_head2(pts_feat)
            # sup_dcn_points_3 = self.sup_dcn_out_points_head3(pts_feat)
            # sup_dcn_offset = torch.cat([sup_dcn_points_1[:,0:6], sup_dcn_points_2[:,0:6], sup_dcn_points_3[:,0:6]], dim=1 )
            # sup_dcn_offset = sup_dcn_points # - self.sup_dcn_base_offset.type_as(sup_dcn_points).to('cuda')
            # sup_dcn_offset_global = sup_dcn_offset + self.sup_dcn_base_offset.type_as(sup_dcn_points).to('cuda')
            # sup_dcn_offset_global_9 = sup_dcn_offset_global[:,0:18]
            # cls_feat = self.spatial_gating_unit(cls_feat)
            # -----end test--------
            # dcn_cls_feat = self.sup_dcn(cls_feat, sup_dcn_offset)
            dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset)
            # 然后继续卷积,目标是分类
            cls_out = self.reppoints_cls_out(self.relu(dcn_cls_feat))
            # 以及,对代表点位置进行进一步微调,reppoints_pts_refine_conv是可形变卷积
            dcn_pts_feat = self.reppoints_pts_refine_conv(pts_feat, dcn_offset)

            # dcn_pts_feat = self.offset_encoder(torch.cat([self.relu(dcn_pts_feat), pts_feature_value], dim=1))
            # dcn_pts_feat = self.offset_encoder2(self.relu(dcn_pts_feat))
            # dcn_pts_feat = dcn_pts_feat + self.offset_encoder3(pts_feature_value)
            pts_out_refine = self.reppoints_pts_refine_out(self.relu(dcn_pts_feat))
            # 微调的结果加上基础值
            pts_out_refine = pts_out_refine + pts_out_init.detach()
            # pts_out_refine = pts_out_refine + core_pts.detach()

            return cls_out, pts_out_init, pts_out_refine, x
            return cls_out, sup_dcn_offset_global_9,pts_out_refine,  x
        
        elif self.my_pts_mode == "drop":
            # TODO test modified
           
            # -------------
            dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset)
            # 然后继续卷积,目标是分类
            cls_out = self.reppoints_cls_out(self.relu(dcn_cls_feat))
            # 以及,对代表点位置进行进一步微调,reppoints_pts_refine_conv是可形变卷积
            pts_out_refine = self.reppoints_pts_refine_out(
                self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
            # 微调的结果加上基础值
            pts_out_refine = pts_out_refine + pts_out_init.detach()
            return cls_out, pts_out_init, pts_out_refine, x

        #
        # 偷天换日,将pts_out_init换成pts_div,仅在test使用,可视化pts_div
        return cls_out, pts_out_init, pts_out_refine, x
        # return cls_out, pts_out_refine,core_pts_grad_temp,  x
    def position_attn(self,feat,position, window_size):
            B, Cannel, W, H = feat.shape
            # override
            C = self.num_points*2
            grid_w = torch.arange(0,W).view(1,1,W,1).repeat(1,1,1,H)
            grid_h = torch.arange(0,H).view(1,1,1,H).repeat(1,1,W,1)
            # TODO  先后顺序？
            # B H W C
            grid = torch.cat([grid_w,grid_h], dim=1).repeat(B,self.num_points,1,1).type_as(feat)
            global_position = grid + position
            global_position = global_position.view(B,H,W,-1)
            att_feat = feat.view(B,H,W,-1)
            # Pading
            window_size = window_size
            pad_l = pad_t = 0
            pad_r = (window_size - W % window_size) % window_size
            pad_b = (window_size - H % window_size) % window_size
            global_position = F.pad(global_position, (0, 0, pad_l, pad_r, pad_t, pad_b))
            att_feat = F.pad(att_feat, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, Hp, Wp, _ = global_position.shape

            # no cyclic shift
            shifted_global_position = global_position
            attn_mask = None

            # partition windows
            global_position_windows = window_partition(shifted_global_position, window_size)  # nW*B, window_size, window_size, C
            global_position_windows = global_position_windows.view(-1, window_size * window_size, C)  # nW*B, window_size*window_size, C
            attn_feat = window_partition(att_feat, window_size)  # nW*B, window_size, window_size, C
            attn_feat = attn_feat.view(-1, window_size * window_size, Cannel)  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            #TODO attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C
            B_, N, C = global_position_windows.shape
            flaten_gp = global_position_windows.view(B_,self.num_points*2,1,-1)
            flaten_gpt = flaten_gp.transpose(-1,-2)
            # res = flaten_gpt @ flaten_gp
            res = flaten_gpt - flaten_gp
            attn = 1-torch.norm(res, p=1, dim=1).clamp(min=1e-12)
            # attn = 1-torch.pow(res, 2)
            # attn = 1-torch .unsqueeze(dim=1)
            attn_windows_out = self.softmax(attn)
            attn_windows_feat = attn_feat.transpose(-1,-2) @ attn_windows_out
            #.transpose(1, 2).reshape(B_, N, C)
            # merge windows
            attn_windows_feat1 = attn_windows_feat.transpose(-1,-2).view(B_, window_size, window_size, Cannel)
            attn_windows_feat2 = window_reverse(attn_windows_feat1, window_size, Hp, Wp)  # B H' W' C
            if pad_r > 0 or pad_b > 0:
                attn_windows_feat2 = attn_windows_feat2[:, :H, :W, :].contiguous()
            # attn_windows_feat2 = attn_windows_feat2.view(B, H * W, Cannel)
            attn_windows_feat2 = attn_windows_feat2.view(B, Cannel,W,H)
            # attn_windows_feat2 = attn_windows_feat2.permute([0,3,2,1])
            return attn_windows_feat2
    
    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_points(self, featmap_sizes, img_metas):
        """
        由图片元数据得到中心候选点的坐标
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)
        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(
                featmap_sizes[i], self.point_strides[i])
            multi_level_points.append(points)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]
        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.point_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w = img_meta['pad_shape'][:2]
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
        return points_list, valid_flag_list

    def offset_to_pts(self, center_list, pred_list):
        """
        由基点坐标和偏移得到全局坐标的代表点
        Args:
            center_list: 基点列表
            pred_list: 自适应点集偏移列表
        Returns: pts_list: 全局坐标系的pts
        """
        pts_list = []
        # 对每个层级分别考虑
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            # 对一个批次中的每个图片分别考虑
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(-1, 2 * self.num_points)
                # 分离横坐标和纵坐标
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                # 基坐标加上偏移得到全局坐标的pts
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    def neargtcorner(self, pts, gtbboxes):
        """
        最近真值角点算法, 对每个自适应点寻找离得最近的gt边界框的角点
        """
        gtbboxes = gtbboxes.view(-1, 4, 2)
        pts = pts.view(-1, self.num_points, 2)

        pts_corner_first_ind = ((gtbboxes[:, 0:1, :] - pts) ** 2).sum(dim=2).min(dim=1)[1]
        pts_corner_first_ind = pts_corner_first_ind.reshape(pts_corner_first_ind.shape[0], 1, 1).expand(-1, -1,
                                                                                                        pts.shape[2])
        pts_corner_first = torch.gather(pts, 1, pts_corner_first_ind).squeeze(1)

        pts_corner_sec_ind = ((gtbboxes[:, 1:2, :] - pts) ** 2).sum(dim=2).min(dim=1)[1]
        pts_corner_sec_ind = pts_corner_sec_ind.reshape(pts_corner_sec_ind.shape[0], 1, 1).expand(-1, -1, pts.shape[2])
        pts_corner_sec = torch.gather(pts, 1, pts_corner_sec_ind).squeeze(1)

        pts_corner_third_ind = ((gtbboxes[:, 2:3, :] - pts) ** 2).sum(dim=2).min(dim=1)[1]
        pts_corner_third_ind = pts_corner_third_ind.reshape(pts_corner_third_ind.shape[0], 1, 1).expand(-1, -1,
                                                                                                        pts.shape[2])
        pts_corner_third = torch.gather(pts, 1, pts_corner_third_ind).squeeze(1)

        pts_corner_four_ind = ((gtbboxes[:, 3:4, :] - pts) ** 2).sum(dim=2).min(dim=1)[1]
        pts_corner_four_ind = pts_corner_four_ind.reshape(pts_corner_four_ind.shape[0], 1, 1).expand(-1, -1,
                                                                                                     pts.shape[2])
        pts_corner_four = torch.gather(pts, 1, pts_corner_four_ind).squeeze(1)

        corners = torch.cat([pts_corner_first, pts_corner_sec, pts_corner_third, pts_corner_four], dim=1)
        return corners

    def sampling_points(self, corners, points_num):
        """
        此方法在由corners组成的四边形的每条边上平均取points_num个点,可以用于计算倒角距离
        Args:
            corners: 角点
            points_num: 每条边取样的点的数目

        Returns:

        """
        device = corners.device
        corners_xs, corners_ys = corners[:, 0::2], corners[:, 1::2]
        # 第一条边的两个点的x和y坐标
        first_edge_x_points = corners_xs[:, 0:2]
        first_edge_y_points = corners_ys[:, 0:2]
        # 第二条边的两个点的x和y坐标
        sec_edge_x_points = corners_xs[:, 1:3]
        sec_edge_y_points = corners_ys[:, 1:3]
        # 第三条边的两个点的x和y坐标
        third_edge_x_points = corners_xs[:, 2:4]
        third_edge_y_points = corners_ys[:, 2:4]
        # 第四条边的两个点的x和y坐标
        four_edge_x_points_s = corners_xs[:, 3]
        four_edge_y_points_s = corners_ys[:, 3]
        four_edge_x_points_e = corners_xs[:, 0]
        four_edge_y_points_e = corners_ys[:, 0]
        # 计算采样比例
        edge_ratio = torch.linspace(0, 1, points_num).to(device).repeat(corners.shape[0], 1)
        # 对第一条边采样10个点
        all_1_edge_x_points = edge_ratio * first_edge_x_points[:, 1:2] + \
                              (1 - edge_ratio) * first_edge_x_points[:, 0:1]
        all_1_edge_y_points = edge_ratio * first_edge_y_points[:, 1:2] + \
                              (1 - edge_ratio) * first_edge_y_points[:, 0:1]
        # 对第二条边采样10个点
        all_2_edge_x_points = edge_ratio * sec_edge_x_points[:, 1:2] + \
                              (1 - edge_ratio) * sec_edge_x_points[:, 0:1]
        all_2_edge_y_points = edge_ratio * sec_edge_y_points[:, 1:2] + \
                              (1 - edge_ratio) * sec_edge_y_points[:, 0:1]
        # 对第三条边采样10个点
        all_3_edge_x_points = edge_ratio * third_edge_x_points[:, 1:2] + \
                              (1 - edge_ratio) * third_edge_x_points[:, 0:1]
        all_3_edge_y_points = edge_ratio * third_edge_y_points[:, 1:2] + \
                              (1 - edge_ratio) * third_edge_y_points[:, 0:1]
        # 对第四条边采样10个点
        all_4_edge_x_points = edge_ratio * four_edge_x_points_e.unsqueeze(1) + \
                              (1 - edge_ratio) * four_edge_x_points_s.unsqueeze(1)
        all_4_edge_y_points = edge_ratio * four_edge_y_points_e.unsqueeze(1) + \
                              (1 - edge_ratio) * four_edge_y_points_s.unsqueeze(1)
        # 将上述结果拼接起来
        all_x_points = torch.cat([all_1_edge_x_points, all_2_edge_x_points,
                                  all_3_edge_x_points, all_4_edge_x_points], dim=1).unsqueeze(dim=2)

        all_y_points = torch.cat([all_1_edge_y_points, all_2_edge_y_points,
                                  all_3_edge_y_points, all_4_edge_y_points], dim=1).unsqueeze(dim=2)

        all_points = torch.cat([all_x_points, all_y_points], dim=2)
        return all_points

    def init_loss_single(self,  pts_pred_init, rbox_gt_init, rbox_weights_init, stride):
        """
        计算初始阶段旋转边界框损失和初始阶段空间约束损失
        Args:
            pts_pred_init:代表点坐标
            rbox_gt_init:真值的旋转边界框
            rbox_weights_init:旋转边界框权重
            stride:系列步长
        Returns:loss_rbox_init, loss_border_init

        """
        normalize_term = self.point_base_scale * stride
        rbox_gt_init = rbox_gt_init.reshape(-1, 8)
        rbox_weights_init = rbox_weights_init.reshape(-1)
        pts_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)
        pos_ind_init = (rbox_weights_init > 0).nonzero().reshape(-1)
        pts_pred_init_norm = pts_pred_init[pos_ind_init]
        rbox_gt_init_norm = rbox_gt_init[pos_ind_init]
        rbox_weights_pos_init = rbox_weights_init[pos_ind_init]
        # 将
        # loss_rbox_init = self.loss_rbox_init(
        #     pts_pred_init_norm / normalize_term,
        #     rbox_gt_init_norm / normalize_term,
        #     rbox_weights_pos_init
        # )
        # loss_border_dist = self.loss_border_dist(
        #     pts_pred_init_norm / normalize_term,
        #     rbox_gt_init_norm / normalize_term
        # )
        loss_border_init = self.loss_spatial_init(
            pts_pred_init_norm.reshape(-1, 2 * self.num_points) / normalize_term,
            rbox_gt_init_norm / normalize_term,
            rbox_weights_pos_init,
            y_first=False,
            avg_factor=None
        ) if self.loss_spatial_init is not None else pts_pred_init.new_zeros(1)

        return loss_border_init
        # return loss_rbox_init, loss_border_init, loss_border_dist

    def loss(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             base_features,
             gt_rbboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_rbboxes_ignore=None):
        # 获取特征图的尺寸
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # 确保二者的层级数目相同
        assert len(featmap_sizes) == len(self.point_generators)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        # 初始阶段的靶目标,由图片各层级尺寸获取各个基点坐标
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas)
        # 由基点坐标和偏移获取全局代表点坐标
        pts_coordinate_preds_init = self.offset_to_pts(center_list,
                                                       pts_preds_init)
        # 每一层级的候选项有多少个
        num_proposals_each_level = [(featmap.size(-1) * featmap.size(-2))
                                    for featmap in cls_scores]
        # 获取层级数目
        num_level = len(featmap_sizes)
        assert num_level == len(pts_coordinate_preds_init)
        candidate_list = center_list

        # 初始阶段 为代表点集分配gt,待分配的点仅仅是中心基点
        cls_reg_targets_init = init_pointset_target(
            candidate_list,
            valid_flag_list,
            gt_rbboxes,
            img_metas,
            cfg.init,
            gt_rbboxes_ignore_list=gt_rbboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        # get the number of sample of assign)
        # 获取分配的样本的数量
        (*_, rbbox_gt_list_init, candidate_list_init, rbox_weights_list_init,
         num_total_pos_init, num_total_neg_init, gt_inds_init) = cls_reg_targets_init

        # 微调阶段
        # 先获取微调后的代表点
        center_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)
        pts_coordinate_preds_refine = self.offset_to_pts(center_list, pts_preds_refine)
        # 获取取样后的代表点的特征向量,9个点即9个特征向量,后面可以计算相似度
        refine_points_features, = multi_apply(self.get_adaptive_points_feature, base_features, pts_coordinate_preds_refine, self.point_strides)
        features_pts_refine_image = levels_to_images(refine_points_features)
        features_pts_refine_image = [item.reshape(-1, self.num_points, item.shape[-1]) for item in features_pts_refine_image]

        points_list = []
        for i_img, center in enumerate(center_list):
            points = []
            for i_lvl in range(len(pts_preds_refine)):
                points_preds_init_ = pts_preds_init[i_lvl].detach()
                points_preds_init_ = points_preds_init_.view(points_preds_init_.shape[0], -1,
                                                             *points_preds_init_.shape[2:])
                # 不同的层级代表不同的步长
                points_shift = points_preds_init_.permute(0, 2, 3, 1) * self.point_strides[i_lvl]
                points_center = center[i_lvl][:, :2].repeat(1, self.num_points)
                # 用基点坐标加上xy偏移
                points.append(points_center + points_shift[i_img].reshape(-1, 2 * self.num_points))
            points_list.append(points)
        # 为微调阶段的代表点集再一次分配gt,待分配的点是init阶段的代表点,即基点加上初始偏移
        cls_reg_targets_refine = refine_pointset_target(
            points_list,
            valid_flag_list,
            gt_rbboxes,
            img_metas,
            cfg.refine,
            gt_rbboxes_ignore_list=gt_rbboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)

        (labels_list, label_weights_list, rbox_gt_list_refine,
         _, rbox_weights_list_refine, pos_inds_list_refine,
         pos_gt_index_list_refine) = cls_reg_targets_refine
        # 将按级别的目标转换为按特征级别的目标
        cls_scores = levels_to_images(cls_scores)
        # 转换目标形状
        cls_scores = [
            item.reshape(-1, self.cls_out_channels) for item in cls_scores
        ]
        # 将按级别的目标转换为按特征级别的目标
        pts_coordinate_preds_init_image = levels_to_images(
            pts_coordinate_preds_init, flatten=True)
        # 转换目标形状
        pts_coordinate_preds_init_image = [
            item.reshape(-1, 2 * self.num_points) for item in pts_coordinate_preds_init_image
        ]
        # 将按级别的目标转换为按特征级别的目标
        pts_coordinate_preds_refine_image = levels_to_images(
            pts_coordinate_preds_refine, flatten=True)
        # 转换目标形状
        pts_coordinate_preds_refine_image = [
            item.reshape(-1, 2 * self.num_points) for item in pts_coordinate_preds_refine_image
        ]

        with torch.no_grad():

            # refine_stage loc loss
            # quality_assess_list, = multi_apply(self.points_quality_assessment, cls_scores,
            #                                pts_coordinate_preds_refine_image, labels_list,
            #                                rbox_gt_list_refine, label_weights_list,
            #                                rbox_weights_list_refine, pos_inds_list_refine)

            # init stage and refine stage loc loss
            # 进行点质量评估, 特色实验
            quality_assess_list, = multi_apply(self.points_quality_assessment, features_pts_refine_image, cls_scores,
                                           pts_coordinate_preds_init_image, pts_coordinate_preds_refine_image, labels_list,
                                           rbox_gt_list_refine, label_weights_list,
                                           rbox_weights_list_refine, pos_inds_list_refine)

            # 根据刚刚得到的各个代表点集的质量, 选择表现最好的点进行取样
            labels_list, label_weights_list, rbox_weights_list_refine, num_pos, pos_normalize_term = multi_apply(
                self.point_samples_selection,
                quality_assess_list,
                labels_list,
                label_weights_list,
                rbox_weights_list_refine,
                pos_inds_list_refine,
                pos_gt_index_list_refine,
                num_proposals_each_level=num_proposals_each_level,
                num_level=num_level
            )
            num_pos = sum(num_pos)
        # 转换目标形状
        cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
        pts_preds_refine = torch.cat(pts_coordinate_preds_refine_image,
                                     0).view(-1, pts_coordinate_preds_refine_image[0].size(-1))
        labels = torch.cat(labels_list, 0).view(-1)
        labels_weight = torch.cat(label_weights_list, 0).view(-1)
        rbox_gt_refine = torch.cat(rbox_gt_list_refine,
                                    0).view(-1, rbox_gt_list_refine[0].size(-1))
        rbox_weights_refine = torch.cat(rbox_weights_list_refine, 0).view(-1)
        pos_normalize_term = torch.cat(pos_normalize_term, 0).reshape(-1)
        pos_inds_flatten = (labels > 0).nonzero().reshape(-1)
        assert len(pos_normalize_term) == len(pos_inds_flatten)
        # 如果存在正样本,则计算各类损失
        if num_pos:
            # 分类损失 采用了focalLoss
            # labels_weight全为1
            # 但是用了index
            losses_cls = self.loss_cls(
                cls_scores, labels, labels_weight, avg_factor=num_pos)
            pos_pts_pred_refine = pts_preds_refine[pos_inds_flatten]
            pos_rbox_gt_refine = rbox_gt_refine[pos_inds_flatten]
            pos_rbox_weights_refine = rbox_weights_refine[pos_inds_flatten]
            # 旋转边界框损失,采用IoULoss
            losses_rbox_refine = self.loss_rbox_refine(
                pos_pts_pred_refine / pos_normalize_term.reshape(-1, 1),
                pos_rbox_gt_refine / pos_normalize_term.reshape(-1, 1),
                pos_rbox_weights_refine
            )
            loss_border_dist_refine = self.loss_border_dist_refine(
                pos_pts_pred_refine / pos_normalize_term.reshape(-1, 1),
                pos_rbox_gt_refine / pos_normalize_term.reshape(-1, 1),
                pos_rbox_weights_refine
            )if self.loss_border_dist_refine is not None else losses_rbox_refine.new_zeros(1)
            # 创新的空间约束损失
            loss_border_refine = self.loss_spatial_refine(
                pos_pts_pred_refine.reshape(-1, 2 * self.num_points) / pos_normalize_term.reshape(-1, 1),
                pos_rbox_gt_refine / pos_normalize_term.reshape(-1, 1),
                pos_rbox_weights_refine,
                y_first=False,
                avg_factor=None
            ) if self.loss_spatial_refine is not None else losses_rbox_refine.new_zeros(1)
            # losses_rbox_init = self.loss_rbox_init(
            #     pos_pts_pred_refine / pos_normalize_term.reshape(-1, 1),
            #     pos_rbox_gt_refine / pos_normalize_term.reshape(-1, 1),
            #     pos_rbox_weights_refine
            # )
            # loss_border_dist = self.loss_border_dist(
            #     pos_pts_pred_refine / pos_normalize_term.reshape(-1, 1),
            #     pos_rbox_gt_refine / pos_normalize_term.reshape(-1, 1),
            #     pos_rbox_weights_refine
            # )if self.loss_border_dist is not None else losses_rbox_refine.new_zeros(1)
            # # loss_border_init = losses_rbox_refine.new_zeros(1)
            # loss_border_init = self.loss_spatial_init(
            #     pos_pts_pred_refine.reshape(-1, 2 * self.num_points) / pos_normalize_term.reshape(-1, 1),
            #     pos_rbox_gt_refine / pos_normalize_term.reshape(-1, 1),
            #     pos_rbox_weights_refine,
            #     y_first=False,
            #     avg_factor=None
            # )if self.loss_spatial_init is not None else losses_rbox_refine.new_zeros(1)

        else:
            losses_cls = cls_scores.sum() * 0
            losses_rbox_refine = pts_preds_refine.sum() * 0
            loss_border_refine = pts_preds_refine.sum() * 0
            loss_border_dist_refine = pts_preds_refine.sum() * 0
            # losses_rbox_init = pts_preds_refine.sum() * 0
            # loss_border_dist= pts_preds_refine.sum() * 0
            # loss_border_init= pts_preds_refine.sum() * 0
        # 计算另外两种损失, 这两种的样本是没有经过质量评估采样的
        # 但是感兴趣的只有gt_num数量的检测
        # SYNC
        # losses_rbox_init, loss_border_init, loss_border_dist = multi_apply(
        # loss_border_init = multi_apply(
            # self.init_loss_single,
            # pts_coordinate_preds_init,
            # rbbox_gt_list_init,
            # rbox_weights_list_init,
            # self.point_strides)
        # END SYNC
         # SYNC


        pts_preds_init = torch.cat(pts_coordinate_preds_init_image,
                                    0).view(-1, pts_coordinate_preds_init_image[0].size(-1))
        rbox_gt_init = torch.cat(rbbox_gt_list_init,
                                0).view(-1, rbbox_gt_list_init[0].size(-1))
        rbox_weights_init = torch.cat(rbox_weights_list_init, 0).view(-1)

        # pos_inds_flatten_init = pos_inds_flatten
        # 还要生成init的pos_normalize_term 
        pos_inds_flatten_init = (rbox_weights_init > 0).nonzero().reshape(-1)
        num_pos_init = len(pos_inds_flatten_init)
        if num_pos_init>0:
            pos_rbox_weights_init = rbox_weights_init[pos_inds_flatten_init]
            # pos_inds_flatten = (labels > 0).nonzero().reshape(-1)
            pos_pts_pred_init = pts_preds_init[pos_inds_flatten_init]
            pos_rbox_gt_init = rbox_gt_init[pos_inds_flatten_init]
            pos_normalize_term_init = self.get_pos_normalize_term(pos_inds_flatten_init,num_proposals_each_level,num_level)
                # 自能改成init h和refine 取样的交集
                # pos_inds_init = (pos_rbox_weights_init > 0).nonzero().reshape(-1)
                # pos_rbox_weights_init = pos_rbox_weights_init[pos_inds_init]
                # pos_normalize_term = pos_normalize_term[pos_inds_init]
                # pos_pts_pred_init = pts_preds_init[pos_inds_init]
                # pos_rbox_gt_init = rbox_gt_init[pos_inds_init]

            losses_rbox_init = self.loss_rbox_init(
                pos_pts_pred_init / pos_normalize_term_init.reshape(-1, 1),
                pos_rbox_gt_init / pos_normalize_term_init.reshape(-1, 1),
                pos_rbox_weights_init
            )
            loss_border_dist = self.loss_border_dist(
                pos_pts_pred_init / pos_normalize_term_init.reshape(-1, 1),
                pos_rbox_gt_init / pos_normalize_term_init.reshape(-1, 1),
                pos_rbox_weights_init
            )if self.loss_border_dist is not None else losses_rbox_refine.new_zeros(1)
            # loss_border_init = losses_rbox_refine.new_zeros(1)
            loss_border_init = self.loss_spatial_init(
                pos_pts_pred_init.reshape(-1, 2 * self.num_points) / pos_normalize_term_init.reshape(-1, 1),
                pos_rbox_gt_init / pos_normalize_term_init.reshape(-1, 1),
                pos_rbox_weights_init,
                y_first=False,
                avg_factor=None
            )if self.loss_spatial_init is not None else losses_rbox_refine.new_zeros(1)

        else:
            losses_rbox_init = pts_preds_refine.sum() * 0
            loss_border_dist= pts_preds_refine.sum() * 0
            loss_border_init= pts_preds_refine.sum() * 0
            # END SYNc
        # 如果不存在正样本,则各类损失为0

        loss_dict_all = {
            'loss_cls': losses_cls,
            'loss_rbox_init': losses_rbox_init,
            'loss_rbox_refine': losses_rbox_refine,
            'loss_spatial_init': loss_border_init,
            'loss_spatial_refine': loss_border_refine,
            'loss_border_dist': loss_border_dist,
            'loss_border_dist_refine': loss_border_dist_refine
        }
        # print(f"loss_cls :{losses_cls};loss_rbox_init:{losses_rbox_init};loss_rbox_refine:{losses_rbox_refine};\
        #       loss_spatial_init:{loss_border_init};loss_spatial_refine:{loss_border_refine};\
        #       loss_border_dist:{loss_border_dist}; loss_border_dist_refine:{loss_border_dist_refine};")
        return loss_dict_all
    
    def get_pos_normalize_term(self, pos_inds_after_select,num_proposals_each_level, num_level):
        num_proposals_each_level_ = num_proposals_each_level.copy()
        num_proposals_each_level_.insert(0, 0)
        inds_level_interval = np.cumsum(num_proposals_each_level_)
        pos_level_mask_after_select = []
        for i in range(num_level):
            mask = (pos_inds_after_select >= inds_level_interval[i]) & (
                    pos_inds_after_select < inds_level_interval[i + 1])
            pos_level_mask_after_select.append(mask)
        pos_level_mask_after_select = torch.stack(pos_level_mask_after_select, 0).type_as(pos_inds_after_select)
        # 由层级掩码和各层级的尺度和步长设置生成正规化系数
        pos_normalize_term = pos_level_mask_after_select * (
                self.point_base_scale *
                torch.as_tensor(self.point_strides).type_as(pos_inds_after_select)).reshape(-1, 1)
        pos_normalize_term = pos_normalize_term[pos_normalize_term > 0].type_as(pos_inds_after_select)
        assert len(pos_normalize_term) == len(pos_inds_after_select)
        return pos_normalize_term

    def get_adaptive_points_feature(self, features, locations, stride):
        '''
        获取自适应点点特征图
        features: [b, c, w, h]
        locations: [b, N, 18]
        stride: the stride of the feature map
        '''

        h = features.shape[2] * stride
        w = features.shape[3] * stride

        locations = locations.view(locations.shape[0], locations.shape[1], -1, 2).clone()
        # 先映射到[0,1],再映射到[-1,1]区间上
        locations[..., 0] = locations[..., 0] / (w / 2.) - 1
        locations[..., 1] = locations[..., 1] / (h / 2.) - 1

        batch_size = features.size(0)
        sampled_features = torch.zeros([locations.shape[0],
                                        features.size(1),
                                        locations.size(1),
                                        locations.size(2)
                                        ]).to(locations.device)
        # 对于一个批次中的每个图片分别处理
        for i in range(batch_size):
            # 对特征他按照网格进行取样, 由于使用的代表点,所以是稀疏的特征向量而不是传统的感兴趣区域
            feature = nn.functional.grid_sample(features[i:i + 1], locations[i:i + 1])[0]
            sampled_features[i] = feature

        return sampled_features,

    def points_quality_assessment(self, points_features, cls_score, pts_pred_init, pts_pred_refine, label, rbbox_gt, label_weight, rbox_weight, pos_inds):
        # 选取正样本pos_inds
        pos_scores = cls_score[pos_inds]
        pos_pts_pred_init = pts_pred_init[pos_inds]
        pos_pts_pred_refine = pts_pred_refine[pos_inds]
        pos_pts_refine_features = points_features[pos_inds]
        # my_qua = self.min_distance_pts(pos_pts_pred_refine)
        pos_rbbox_gt = rbbox_gt[pos_inds]
        pos_label = label[pos_inds]
        pos_label_weight = label_weight[pos_inds]
        pos_rbox_weight = rbox_weight[pos_inds]
        # 计算特征向量相似度
        pts_feats_dissimilarity = self.feature_cosine_similarity(pos_pts_refine_features)
        # 计算分类损失
        # weight的作用, 分类的全为1,rbox的只有正样本为1
        qua_cls = self.loss_cls(
            pos_scores,
            pos_label,
            pos_label_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')
        # 计算最小面积矩形
        corners_pred_init = minaerarect(pos_pts_pred_init)
        corners_pred_refine = minaerarect(pos_pts_pred_refine)
        # corners_pred = self.neargtcorner(pos_pts_pred, pos_rbbox_gt)

        # 此方法在由corners组成的四边形的每条边上平均取points_num个点, 可以用于计算倒角距离
        sampling_pts_pred_init = self.sampling_points(corners_pred_init, 10)
        sampling_pts_pred_refine = self.sampling_points(corners_pred_refine, 10)
        corners_pts_gt = self.sampling_points(pos_rbbox_gt, 10)

        # 计算倒角距离
        qua_ori_init = ChamferDistance2D(corners_pts_gt, sampling_pts_pred_init)
        qua_ori_refine = ChamferDistance2D(corners_pts_gt, sampling_pts_pred_refine)
        # 计算旋转边界框损失
        qua_loc_init = self.loss_rbox_refine(
            pos_pts_pred_init,
            pos_rbbox_gt,
            pos_rbox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')
        # 计算旋转边界框损失
        qua_loc_refine = self.loss_rbox_refine(
            pos_pts_pred_refine,
            pos_rbbox_gt,
            pos_rbox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        qua_cls = qua_cls.sum(-1)
        # weight inti-stage and refine-stage
        # TODO
        # qua = qua_cls + 0.2*(qua_loc_init + 0.3 * qua_ori_init) + 0.8 * (
        #             qua_loc_refine + 0.3 * qua_ori_refine) + 0.1*pts_feats_dissimilarity
        # 分配各项的权重
        # 为了保持权重一致所以都用refine损失函数
        loss_border_dist_init = self.loss_border_dist_refine(
            pos_pts_pred_init,
            pos_rbbox_gt,
            pos_rbox_weight,
            reduction_override='none'
        )if self.loss_border_dist_refine is not None else qua_loc_refine.new_zeros(1)

        loss_border_dist_refine = self.loss_border_dist_refine(
            pos_pts_pred_refine,
            pos_rbbox_gt,
            pos_rbox_weight,
            reduction_override='none'
        )if self.loss_border_dist_refine is not None else qua_loc_refine.new_zeros(1)
        qua = qua_cls + 0.2 * (qua_loc_init + 0.3 * qua_ori_init + 1.* loss_border_dist_init) + 0.8 * (
                qua_loc_refine + 0.3 * qua_ori_refine + 1. *loss_border_dist_refine) #+ 0.1 * pts_feats_dissimilarity

        # qua = qua_cls + 1.2 * (qua_loc_init + 0.3 * qua_ori_init + 1.* loss_border_dist_init) + 1.8 * (
        #         qua_loc_refine + 0.3 * qua_ori_refine + 1. *loss_border_dist_refine) #+ 0.1 * pts_feats_dissimilarity

        return qua,

    # def min_distance_pts(self, pts):
    #     if pts.shape[0] == 0:
    #         return 0
    #     pts_pred = pts.reshape(-1, self.num_points, 2)
    #     p1 = torch.unsqueeze(pts_pred, dim=1)
    #     p2 = torch.unsqueeze(pts_pred, dim=2)
    #     distance = torch.sum((p1 - p2)**2, dim=-1)
    #     di2 = distance
    #     eye9 = np.eye(9)
    #     et9 = torch.eye(9)
    #     no_diag = distance[:, torch.where(et9 == 0)[0], torch.where(et9 == 0)[1]]
    #     min, ind = torch.min(no_diag, dim= 1)
    #     max, ind = torch.max(no_diag, dim= 1)
    #     return torch.div(1, min+1.)

    def feature_cosine_similarity(self, points_features):
        '''
        使用余弦相似度计算各自适应点的特征向量的相似度
        points_features: [N_pos, 9, 256]
        '''
        # print('points_features', points_features.shape)
        mean_points_feats = torch.mean(points_features, dim=1, keepdim=True)
        # print('mean_points_feats', mean_points_feats.shape)

        norm_pts_feats = torch.norm(points_features, p=2, dim=2).unsqueeze(dim=2).clamp(min=1e-2)
        # print('norm_pts_feats', norm_pts_feats)
        norm_mean_pts_feats = torch.norm(mean_points_feats, p=2, dim=2).unsqueeze(dim=2).clamp(min=1e-2)
        # print('norm_mean_pts_feats', norm_mean_pts_feats)

        unity_points_features = points_features / norm_pts_feats

        # 正则化特征向量
        unity_mean_points_feats = mean_points_feats / norm_mean_pts_feats
        # 计算余弦相似度
        cos_similarity = nn.CosineSimilarity(dim=2, eps=1e-6)
        # 特征向量相似度
        feats_similarity = 1.0 - cos_similarity(unity_points_features, unity_mean_points_feats)
        # print('feats_similarity', feats_similarity)

        # TODO
        if len(feats_similarity) == 0:
            max_correlation = 0
        else:
            max_correlation, _ = torch.max(feats_similarity, dim=1)

        # 取最大值
        # max_correlation, _ = torch.max(feats_similarity, dim=1)
        # print('max_correlation', max_correlation.shape, max_correlation)
        return max_correlation

    def point_samples_selection(self, quality_assess, label, label_weight, rbox_weight,
                     pos_inds, pos_gt_inds, num_proposals_each_level=None, num_level=None):
        '''
              基于质量评估值的代表点集选择
        '''

        if len(pos_inds) == 0:
            return label, label_weight, rbox_weight, 0, torch.tensor([]).type_as(rbox_weight)

        num_gt = pos_gt_inds.max()
        num_proposals_each_level_ = num_proposals_each_level.copy()
        num_proposals_each_level_.insert(0, 0)
        inds_level_interval = np.cumsum(num_proposals_each_level_)
        pos_level_mask = []
        # 对于各个层级分别考虑,生成level_mask
        for i in range(num_level):
            mask = (pos_inds >= inds_level_interval[i]) & (
                    pos_inds < inds_level_interval[i + 1])
            pos_level_mask.append(mask)
        pos_inds_after_select = []
        ignore_inds_after_select = []

        for gt_ind in range(num_gt):
            pos_inds_select = []
            pos_loss_select = []
            # gt_mask 一次之考虑匹配一个gt的检测
            gt_mask = pos_gt_inds == (gt_ind + 1)
            for level in range(num_level):
                # level_mask 一次只考虑一个层级
                level_mask = pos_level_mask[level]
                level_gt_mask = level_mask & gt_mask
                # topk largest=False 从小到大,每个gt每层最多6个gt选出
                # 每层最多选出6个,所有层级选完之后还有再进行选择
                value, topk_inds = quality_assess[level_gt_mask].topk(
                    min(level_gt_mask.sum(), 6), largest=False)
                pos_inds_select.append(pos_inds[level_gt_mask][topk_inds])
                pos_loss_select.append(value)
            pos_inds_select = torch.cat(pos_inds_select)
            pos_loss_select = torch.cat(pos_loss_select)

            # 如果选出了少于2个,那么就不用再选了
            if len(pos_inds_select) < 2:
                pos_inds_after_select.append(pos_inds_select)
                ignore_inds_after_select.append(pos_inds_select.new_tensor([]))
            else:
                # 否则继续根据质量从小到大排序,按照一定比例取样,这个比例是个超参数
                pos_loss_select, sort_inds = pos_loss_select.sort() # small to large
                pos_inds_select = pos_inds_select[sort_inds]
                topk = math.ceil(pos_loss_select.shape[0] * self.top_ratio)
                pos_inds_select_topk = pos_inds_select[:topk]
                pos_inds_after_select.append(pos_inds_select_topk)
                ignore_inds_after_select.append(pos_inds_select_topk.new_tensor([]))

        pos_inds_after_select = torch.cat(pos_inds_after_select)
        ignore_inds_after_select = torch.cat(ignore_inds_after_select)
        # 判断有没有被取样,A != B返回一个布尔型矩阵,而all()则表示一行中有一个为真即可
        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_select).all(1)
        reassign_ids = pos_inds[reassign_mask]
        # 没有被取样的点给归零
        label[reassign_ids] = 0
        # label[reassign_ids] = self.num_classes
        label_weight[ignore_inds_after_select] = 0
        rbox_weight[reassign_ids] = 0
        num_pos = len(pos_inds_after_select)

        # 为取样后的代表点生成层级掩码
        pos_level_mask_after_select = []
        for i in range(num_level):
            mask = (pos_inds_after_select >= inds_level_interval[i]) & (
                    pos_inds_after_select < inds_level_interval[i + 1])
            pos_level_mask_after_select.append(mask)
        pos_level_mask_after_select = torch.stack(pos_level_mask_after_select, 0).type_as(label)
        # 由层级掩码和各层级的尺度和步长设置生成正规化系数
        pos_normalize_term = pos_level_mask_after_select * (
                self.point_base_scale *
                torch.as_tensor(self.point_strides).type_as(label)).reshape(-1, 1)
        pos_normalize_term = pos_normalize_term[pos_normalize_term > 0].type_as(rbox_weight)
        assert len(pos_normalize_term) == len(pos_inds_after_select)

        return label, label_weight, rbox_weight, num_pos, pos_normalize_term

    def get_bboxes(self,
                   cls_scores,
                   pts_preds_init,
                   pts_preds_refine,
                   base_feats,
                   img_metas,
                   cfg,
                   rescale=False,
                   nms=True):
        assert len(cls_scores) == len(pts_preds_refine)
        
        num_levels = len(cls_scores)
        mlvl_points = [
            self.point_generators[i].grid_points(cls_scores[i].size()[-2:],
                                                 self.point_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            # 分类分数列表
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            # 自适应点偏移列表
            points_pred_list = [
                pts_preds_refine[i][img_id].detach()
                for i in range(num_levels)
            ]
            points_init_list = [
                pts_preds_init[i][img_id].detach()
                for i in range(num_levels)
            ]
            # 原始图片形状
            img_shape = img_metas[img_id]['img_shape']
            # 尺度系数,用于还原到原始图片尺寸
            scale_factor = img_metas[img_id]['scale_factor']
            # TODO 现在我想把pts_init一并输出展示
            proposals = self.get_bboxes_single(cls_score_list, points_pred_list, points_init_list,
                                               mlvl_points, img_shape,
                                               scale_factor, cfg, rescale, nms)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          points_preds,
                          points_inits,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False,
                          nms=True):
        assert len(cls_scores) == len(points_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_bboxes_init = []
        mlvl_scores = []
        mlvl_reppoints = []
        mlvl_reppoints_init = []

        for i_lvl, (cls_score, points_pred, points_init, points) in enumerate(
                zip(cls_scores, points_preds, points_inits, mlvl_points)):
            assert cls_score.size()[-2:] == points_pred.size()[-2:]
            # 交换维度顺序, 然后改变形状
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            # 如果不使用sigmoid分类,则第0位为背景类,需要注意
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            # 交换维度顺序, 然后改变形状
            points_pred = points_pred.permute(1, 2, 0).reshape(-1, 2 * self.num_points)
            points_init = points_init.permute(1, 2, 0).reshape(-1, 2 * self.num_points)
            # 如果配置文件中没有,则置nms_pre为-1
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # 如果不使用sigmoid分类,则第0位为背景类,需要注意
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                # 选择分数最大的nms_pre个代表点集
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                points_pred = points_pred[topk_inds, :]
                points_init = points_init[topk_inds, :]
                scores = scores[topk_inds, :]
            # 变换代表点偏移的形状
            pts_pred = points_pred.reshape(-1, self.num_points, 2)
            pts_pred_offsety = pts_pred[:, :, 0::2]
            pts_pred_offsetx = pts_pred[:, :, 1::2]
            pts_pred = torch.cat([pts_pred_offsetx, pts_pred_offsety], dim=2).reshape(-1, 2 * self.num_points)
            # 使用最小面积矩形方法由代表点变换为定向矩形边界框
            bbox_pred = minaerarect(pts_pred)
            # 由基点坐标,将定向矩形边界框转换为全局坐标系上
            bbox_pos_center = points[:, :2].repeat(1, 4)
            bboxes = bbox_pred * self.point_strides[i_lvl] + bbox_pos_center
            
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

            # 变换代表点偏移的形状
            points_pred = points_pred.reshape(-1, self.num_points, 2)
            points_pred_dy = points_pred[:, :, 0::2]
            points_pred_dx = points_pred[:, :, 1::2]
            pts = torch.cat([points_pred_dx, points_pred_dy], dim=2).reshape(-1, 2 * self.num_points)
            # 由基点坐标,将代表点偏移转换为全局坐标系上
            pts_pos_center = points[:, :2].repeat(1, self.num_points)
            pts = pts * self.point_strides[i_lvl] + pts_pos_center

            mlvl_reppoints.append(pts)

            # 新增init 点
            points_init = points_init.reshape(-1, self.num_points, 2)
            points_init_dy = points_init[:, :, 0::2]
            points_init_dx = points_init[:, :, 1::2]
            pts_init = torch.cat([points_init_dx, points_init_dy], dim=2).reshape(-1, 2 * self.num_points)
            # 由基点坐标,将代表点偏移转换为全局坐标系上
            pts_pos_center = points[:, :2].repeat(1, self.num_points)
            pts_init = pts_init * self.point_strides[i_lvl] + pts_pos_center

            mlvl_reppoints_init.append(pts_init)

        # 拼接多个Tensor
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_reppoints = torch.cat(mlvl_reppoints)
        mlvl_reppoints_init = torch.cat(mlvl_reppoints_init)
        # 如果rescale, 则将各种全局坐标还原到原始图片尺寸上
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_reppoints /= mlvl_reppoints.new_tensor(scale_factor)
            mlvl_reppoints_init /= mlvl_reppoints_init.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        # 对每个类别应用非极大值抑制
        # TODO
        if nms:
            # det_bboxes, det_labels = multiclass_rnms(mlvl_bboxes, mlvl_scores,cfg.score_thr, cfg.nms,cfg.max_per_img, multi_reppoints=mlvl_reppoints)
            det_bboxes, det_labels = multiclass_rnms(mlvl_bboxes, mlvl_scores,cfg.score_thr, cfg.nms,cfg.max_per_img, multi_reppoints=mlvl_reppoints,multi_reppoints_init = mlvl_reppoints_init)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores


