import torch
import torch.nn as nn
from mmdet.models.utils import ConvModule

class BasicRFB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.conv_cfg = None
        # self.norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        # self.norm_cfg = dict(type='GN', num_groups=16, requires_grad=True)
        # self.norm_cfg = dict(type='SyncBN')
        self.norm_cfg = dict(type='BN')
        self.branch0 = nn.Sequential(
            ConvModule(in_planes, 2*inter_planes, kernel_size=1, stride=stride,conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None),
            ConvModule(2*inter_planes, 2 * inter_planes, kernel_size=(3,1), stride=1, padding=(visual,0), dilation=visual,
                        conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None, activation=None)
            )
        self.branch1 = nn.Sequential(
            ConvModule(in_planes, inter_planes, kernel_size=1, stride=stride, conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg, bias=self.norm_cfg is None),
            ConvModule(inter_planes, 2 * inter_planes, kernel_size=(3,1), stride=stride, padding=(visual,0),
                       conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None),
            ConvModule(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=(visual+1, 1),dilation=(visual+1,1),
                       conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None,activation=None)
            )
        self.branch2 = nn.Sequential(
            ConvModule(in_planes, inter_planes, kernel_size=1, stride=stride, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None),
            ConvModule(inter_planes, (inter_planes//2)*3, kernel_size=(3, 1), stride=stride, padding=(visual, 0), conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None),
            ConvModule((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1 ,conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None),
            ConvModule(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=(2*visual + 1, 1),dilation=(2*visual+1,1),
                       conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None,activation=None)
            )

        self.ConvLinear = ConvModule(6*inter_planes, out_planes, kernel_size=1, stride=1,
                                    conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None, activation=None)
        self.shortcut = ConvModule(in_planes, out_planes, kernel_size=1, stride=1,
                                    conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, bias=self.norm_cfg is None, activation=None)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)
        return out