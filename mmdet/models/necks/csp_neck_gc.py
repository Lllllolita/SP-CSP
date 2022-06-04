import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from mmdet.ops.gcb import ContextBlock
from ..registry import NECKS
from ..utils import ConvModule
import cv2

@NECKS.register_module
class CSPNeckGC(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(CSPNeckGC, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        # assert self.num_ins == 3
        assert self.num_ins == 4
        self.num_outs = num_outs
        self.activation = activation
        self.fp16_enabled = False

        # self.p3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        # self.p4 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=4, padding=0)
        # self.p5 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=4, padding=0)
        self.p3 = nn.ConvTranspose2d(in_channels[1], in_channels[1], kernel_size=4, stride=2, padding=1)
        self.p4 = nn.ConvTranspose2d(in_channels[2], in_channels[2], kernel_size=4, stride=4, padding=0)
        self.p5 = nn.ConvTranspose2d(in_channels[3], in_channels[3], kernel_size=8, stride=8, padding=0)
        #
        # self.p3_l2 = L2Norm(256, 10)
        # self.p4_l2 = L2Norm(256, 10)
        # self.p5_l2 = L2Norm(256, 10)
        self.gcblock = nn.ModuleList([ContextBlock(in_channels[0], ratio = 1/16, pooling_type='att', fusion_types=('channel_add',)),
                                      ContextBlock(in_channels[1], ratio = 1/16, pooling_type='att', fusion_types=('channel_add',)),
                                      ContextBlock(in_channels[2], ratio = 1/16, pooling_type='att', fusion_types=('channel_add',)),
                                      ContextBlock(in_channels[3], ratio = 1/16, pooling_type='att', fusion_types=('channel_add',))])

        self.p2_l2 = L2Norm(in_channels[0], 10)
        self.p3_l2 = L2Norm(in_channels[1], 10)
        self.p4_l2 = L2Norm(in_channels[2], 10)
        self.p5_l2 = L2Norm(in_channels[3], 10)

        self.fusion = nn.Conv2d(sum(in_channels), out_channels, kernel_size=1)
    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         xavier_init(m, distribution='uniform')
        xavier_init(self.p3, distribution='normal')
        xavier_init(self.p4, distribution='normal')
        xavier_init(self.p5, distribution='normal')
        xavier_init(self.fusion, distribution='normal')

    def normalize(self, x):
        x = x - x.min()
        x = x/x.max()
        return x

    def feature_map_visualization(self, x, y):
        x = x[0].detach().cpu().numpy()
        y = y[0].detach().cpu().numpy()
        first = self.normalize(x[0])
        second = self.normalize(y[0])
        cv2.imshow('1', first)
        cv2.imshow('2', second)
        cv2.waitKey(0)

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        f2, f3, f4, f5 = inputs

        g2 = self.gcblock[0](f2)
        g3 = self.gcblock[1](f3)
        g4 = self.gcblock[2](f4)
        g5 = self.gcblock[3](f5)

        p2 = f2 + g2
        f3 = f3 + g3
        f4 = f4 + g4
        f5 = f5 + g5

        p3 = self.p3(f3)
        p4 = self.p4(f4)
        p5 = self.p5(f5)

        p2 = self.p2_l2(p2)
        p3 = self.p3_l2(p3)
        p4 = self.p4_l2(p4)
        p5 = self.p5_l2(p5)

        cat = torch.cat([p2, p3, p4, p5], dim=1)
        # outs=[cat]
        fusion = self.fusion(cat)
        outs=[fusion]
        return tuple(outs)

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out