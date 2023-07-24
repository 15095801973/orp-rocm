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
import math
import matplotlib.pyplot as plt


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
                 loss_border_dist_refine = dict(type='BorderDistLoss', loss_weight=0.8)
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
        self.loss_border_dist = build_loss(loss_border_dist_init)
        self.loss_border_dist_refine = build_loss(loss_border_dist_refine)
        self.drop = nn.Dropout(0.1)

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
        self.sup_num_points = 25
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
        # TODO 添加自适应点,专门用来分类
        if self.my_pts_mode == "pts_down" or self.my_pts_mode == "pts_up":
            self.div_reppoints_conv = nn.Conv2d(self.feat_channels,
                                                     self.point_feat_channels, 3,
                                                     1, 1)
            # 以上一层的输出为输入,输出了代表点,也是可形变卷积的xy偏移
            self.div_reppoints_point = nn.Conv2d(self.point_feat_channels,
                                                    pts_out_dim, 1, 1, 0)
        if self.my_pts_mode == "int":
            self.div_common_conv1 = nn.Conv2d(self.feat_channels,
                                              self.point_feat_channels, 1, 1, 0)
        if self.my_pts_mode == "com1":
            self.div_common_conv1 = nn.Conv2d(self.feat_channels,
                                              self.point_feat_channels, 1, 1, 0)
        if self.my_pts_mode == "com3":
            self.div_common_conv1 = nn.Conv2d(self.feat_channels,
                                              self.point_feat_channels, 3, 1, 1)
        if self.my_pts_mode == "com5":
            self.div_common_conv1 = nn.Conv2d(self.feat_channels,
                                              self.point_feat_channels, 5, 1, 2)
        if self.my_pts_mode == "demo" or self.my_pts_mode == "pts_down" or \
            self.my_pts_mode == "pts_up" or self.my_pts_mode == "int" or self.my_pts_mode == "drop":
            self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
        # todo
        # self.reppoints_cls_conv = DeformConv(self.feat_channels,
        #                                      self.point_feat_channels,
        #                                      self.dcn_kernel, 1, self.dcn_pad)
        # self.conv1 = nn.Conv2d(self.feat_channels,
        #                                       self.point_feat_channels, 1, 1, 0)
        # self.conv3 = nn.Conv2d(self.feat_channels,
                                            #   self.point_feat_channels, 3, 1, 1)
        # self.conv5 = nn.Conv2d(self.feat_channels,
                                            #   self.point_feat_channels, 5, 1, 2)
        if self.my_pts_mode == "ide3":
            self.conv_ide3 = nn.Conv2d(self.feat_channels,
                                              self.point_feat_channels, 3, 1, 1)
        if self.my_pts_mode == "core":
            # 注意1x1dcn的pad设置为0, 但是1x1源文件中有问题, 还是用伪3x3吧
            # self.core_dcn = DeformConv(self.feat_channels, self.point_feat_channels, 3, 1, 1)
            self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                                 self.point_feat_channels,
                                                 self.dcn_kernel, 1, self.dcn_pad)
        if self.my_pts_mode == "sup_dcn":
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
        if self.my_pts_mode == "core_v2" or self.my_pts_mode == "core_v3":
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
        # if self.my_pts_mode == "com3" or self.my_pts_mode == "com1":
        if self.my_pts_mode[0:3] == "com":
                normal_init(self.div_common_conv1, std=0.01)
        if self.my_pts_mode == "int":
                normal_init(self.div_common_conv1, std=0.01)
        # else:
        if self.my_pts_mode == "demo" or self.my_pts_mode == "pts_down" or self.my_pts_mode == "pts_up" or self.my_pts_mode == "int" or self.my_pts_mode == "drop":
            normal_init(self.reppoints_cls_conv, std=0.01)
        
        if self.my_pts_mode == "ide3":
            normal_init(self.conv_ide3, std=0.01)
        
        if self.my_pts_mode == "sup_dcn":
            normal_init(self.sup_dcn, std=0.01)
            normal_init(self.sup_dcn_conv, std=0.01)
            normal_init(self.sup_dcn_out, std=0.01)

        if self.my_pts_mode == "core":
            normal_init(self.reppoints_cls_conv, std=0.01)
        if self.my_pts_mode == "core_v2" or self.my_pts_mode == "core_v3":
            normal_init(self.reppoints_cls_conv, std=0.01)
            normal_init(self.pseudo_dcn_pts, std=0.01)
            normal_init(self.pseudo_dcn_cls, std=0.01)
        # normal_init(self.conv1, std=0.01)
        # normal_init(self.conv3, std=0.01)
        # todo
        # normal_init(self.reppoints_cls_conv, std=0.01)

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
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
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

        # end
        # 细化并且对代表点分类
        # gradient_mul 反向传播梯度的因子,作用是控制反向传播时的比例
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        # copy
        if self.my_pts_mode == "pts_down" or self.my_pts_mode == "pts_up":
            # pts_div_grad_mul = (1 - self.gradient_mul) * pts_div.detach() + self.gradient_mul * pts_out_init
            # pts_div_offset = pts_div_grad_mul - dcn_base_offset
            # TODO 你不是全为零吗,满足你
            # pts_div = dcn_offset.new_zeros(dcn_offset.shape) #+ dcn_base_offset
            # end

            pts_div_offset = pts_div - dcn_base_offset
        
            
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
        
        elif self.my_pts_mode == "core_v2":

            # 改正一个错误, core应该对pts进行而不是offset
            # 因为dcn_base_offset有偏移， 否则是中心开花而不是单点
            alter = True
            if alter == True:
                pts_x_mean = pts_out_init[:, 0::2].mean(dim=1).unsqueeze(1)
                pts_y_mean = pts_out_init[:, 1::2].mean(dim=1).unsqueeze(1)
                core_pts = torch.cat([pts_x_mean, pts_y_mean], dim=1).repeat(1, 9, 1, 1)
                core_pts_grad_temp = (1 - self.gradient_mul) * core_pts.detach() + self.gradient_mul * core_pts
                core_offset = core_pts_grad_temp - dcn_base_offset
                # core_offset =  torch.zeros_like(core_pts_grad_temp)- dcn_base_offset
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

            # 对照实验
            after_date = 713
            if after_date == 713:
                # 附带余弦相似度的修改line917
                dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset) + self.pseudo_dcn_cls(cls_feat, core_offset) / torch.tensor(9).type_as(x)
            elif after_date == 712:
                dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset)
            else:
                dcn_cls_feat = self.reppoints_cls_conv(cls_feat, core_offset)
            
            # dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset)

            cls_out = self.reppoints_cls_out(self.relu(dcn_cls_feat))
            
            dcn_temp = self.reppoints_pts_refine_conv(pts_feat, dcn_offset)
            # 伪-单点可形变卷积
            pts_refine_temp = dcn_temp  + self.pseudo_dcn_pts(pts_feat, core_offset) / torch.tensor(9).type_as(x)# pts_feat  # + self.conv3(pts_feat)+ self.conv1(pts_feat) 
            pts_out_refine = self.reppoints_pts_refine_out(self.relu(pts_refine_temp))
            pts_out_refine = pts_out_refine + pts_out_init.detach()
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
            cls_out = self.reppoints_cls_out(self.conv1(cls_feat* torch.tensor(self.num_points).type_as(x)))

            dcn_temp = self.reppoints_pts_refine_conv(pts_feat, dcn_offset)
            # norefine_temp = dcn_temp + self.conv1(pts_feat) + pts_feat  # + self.conv3(pts_feat)
            pts_out_refine = self.reppoints_pts_refine_out(self.relu(dcn_temp))
            pts_out_refine = pts_out_refine + pts_out_init.detach()

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
        elif self.my_pts_mode == "demo":
            # TODO test modified
            # pts_x_mean = pts_out_init[:, 0::2].mean(dim=1, keepdim=True)
            # pts_y_mean = pts_out_init[:, 1::2].mean(dim=1, keepdim=True)
            # core_pts = torch.cat([pts_x_mean, pts_y_mean], dim=1).repeat(1, 9, 1, 1).detach()
            # pts_grad_temp = (1 - self.gradient_mul) * core_pts.detach() + self.gradient_mul * core_pts
            # core_offset = pts_grad_temp - dcn_base_offset
            # a_cls_feat = self.reppoints_cls_conv(cls_feat, core_offset)
            # a_cls_out = self.reppoints_cls_out(self.relu(a_cls_feat))
            # -----end test--------
            dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset)
            # 然后继续卷积,目标是分类
            cls_out = self.reppoints_cls_out(self.relu(dcn_cls_feat))
            # 以及,对代表点位置进行进一步微调,reppoints_pts_refine_conv是可形变卷积
            pts_out_refine = self.reppoints_pts_refine_out(
                self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
            # 微调的结果加上基础值
            pts_out_refine = pts_out_refine + pts_out_init.detach()
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
        #
        # 偷天换日,将pts_out_init换成pts_div,仅在test使用,可视化pts_div
        return cls_out, pts_out_init, pts_out_refine, x
        # return cls_out, pts_out_refine,core_pts_grad_temp,  x

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
           
        else:
            losses_cls = cls_scores.sum() * 0
            losses_rbox_refine = pts_preds_refine.sum() * 0
            loss_border_refine = pts_preds_refine.sum() * 0
            loss_border_dist_refine = pts_preds_refine.sum() * 0
            
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
        )

        loss_border_dist_refine = self.loss_border_dist_refine(
            pos_pts_pred_refine,
            pos_rbbox_gt,
            pos_rbox_weight,
            reduction_override='none'
        )
        qua = qua_cls + 0.2 * (qua_loc_init + 0.3 * qua_ori_init + 1.* loss_border_dist_init) + 0.8 * (
                qua_loc_refine + 0.3 * qua_ori_refine + 1. *loss_border_dist_refine) + 0.1 * pts_feats_dissimilarity

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


