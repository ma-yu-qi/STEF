# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.model import BaseModule, ModuleList
from torch import nn

from mmocr.registry import MODELS

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.
  
class eSEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel,channel, kernel_size=1,
                             padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x

class FPEM(BaseModule):
    """FPN-like feature fusion module in PANet.

    Args:
        in_channels (int): Number of input channels.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels: int = 128,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.up_add1 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add2 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add3 = SeparableConv2d(in_channels, in_channels, 1)
        self.down_add1 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add2 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add3 = SeparableConv2d(in_channels, in_channels, 2)

    def forward(self, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor,
                c5: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            c2, c3, c4, c5 (Tensor): Each has the shape of
                :math:`(N, C_i, H_i, W_i)`.

        Returns:
            list[Tensor]: A list of 4 tensors of the same shape as input.
        """
        # upsample
        c4 = self.up_add1(self._upsample_add(c5, c4))  # c4 shape
        c3 = self.up_add2(self._upsample_add(c4, c3))
        c2 = self.up_add3(self._upsample_add(c3, c2))

        # downsample
        c3 = self.down_add1(self._upsample_add(c3, c2))
        c4 = self.down_add2(self._upsample_add(c4, c3))
        c5 = self.down_add3(self._upsample_add(c5, c4))  # c4 / 2
        return c2, c3, c4, c5

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) + y


class SeparableConv2d(BaseModule):
    """Implementation of separable convolution, which is consisted of depthwise
    convolution and pointwise convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the depthwise convolution.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            groups=in_channels)
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


@MODELS.register_module()
class FPEM_FFMese(BaseModule):
    """This code is from https://github.com/WenmuZhou/PAN.pytorch.

    Args:
        in_channels (list[int]): A list of 4 numbers of input channels.
        conv_out (int): Number of output channels.
        fpem_repeat (int): Number of FPEM layers before FFM operations.
        align_corners (bool): The interpolation behaviour in FFM operation,
            used in :func:`torch.nn.functional.interpolate`.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(
        self,
        in_channels: List[int],
        conv_out: int = 128,
        fpem_repeat: int = 2,
        align_corners: bool = False,
        init_cfg: Optional[Union[Dict, List[Dict]]] = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        # reduce layers
        self.reduce_conv_c2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[0],
                out_channels=conv_out,
                kernel_size=1), nn.BatchNorm2d(conv_out), nn.ReLU())
        self.reduce_conv_c3 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[1],
                out_channels=conv_out,
                kernel_size=1), nn.BatchNorm2d(conv_out), nn.ReLU())
        self.reduce_conv_c4 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[2],
                out_channels=conv_out,
                kernel_size=1), nn.BatchNorm2d(conv_out), nn.ReLU())
        self.reduce_conv_c5 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[3],
                out_channels=conv_out,
                kernel_size=1), nn.BatchNorm2d(conv_out), nn.ReLU())
        self.align_corners = align_corners
        self.fpems = ModuleList()
        for _ in range(fpem_repeat):
            self.fpems.append(FPEM(conv_out))
        
        self.eSE = eSEModule(conv_out*4) # eSE模块

    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        """
        Args:
            x (list[Tensor]): A list of four tensors of shape
                :math:`(N, C_i, H_i, W_i)`, representing C2, C3, C4, C5
                features respectively. :math:`C_i` should matches the number in
                ``in_channels``.

        Returns:
            tuple[Tensor]: Four tensors of shape
            :math:`(N, C_{out}, H_0, W_0)` where :math:`C_{out}` is
            ``conv_out``.
        """
        c2, c3, c4, c5 = x
        # reduce channel
        c2 = self.reduce_conv_c2(c2)
        c3 = self.reduce_conv_c3(c3)
        c4 = self.reduce_conv_c4(c4)
        c5 = self.reduce_conv_c5(c5)

        # FPEM
        for i, fpem in enumerate(self.fpems):
            c2, c3, c4, c5 = fpem(c2, c3, c4, c5)
            if i == 0:
                c2_ffm = c2
                c3_ffm = c3
                c4_ffm = c4
                c5_ffm = c5
            else:
                c2_ffm = c2_ffm + c2
                c3_ffm = c3_ffm + c3
                c4_ffm = c4_ffm + c4
                c5_ffm = c5_ffm + c5

        # FFM
        c5 = F.interpolate(
            c5_ffm,
            c2_ffm.size()[-2:],
            mode='bilinear',
            align_corners=self.align_corners)
        c4 = F.interpolate(
            c4_ffm,
            c2_ffm.size()[-2:],
            mode='bilinear',
            align_corners=self.align_corners)
        c3 = F.interpolate(
            c3_ffm,
            c2_ffm.size()[-2:],
            mode='bilinear',
            align_corners=self.align_corners)
        #outs = [c2_ffm, c3, c4, c5]
                # 打印每个输出张量的通道数
        # for i, out_tensor in enumerate(outs, start=2):
        #     print(f"Channels in output {i}: {out_tensor.size(1)}")
        #return tuple(outs)
        # 将四个张量沿着通道维度连接起来
        out = torch.cat([c2_ffm, c3, c4, c5], dim=1)
        out = self.eSE(out)
        return out

