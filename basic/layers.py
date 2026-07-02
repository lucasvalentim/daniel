
#  Copyright Université de Rouen Normandie (1), tutelle du laboratoire LITIS (1)
#  contributors :
#  - Denis Coquenet
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

import math
import random

import torch
from torch.nn import Conv2d, Dropout, Dropout2d, InstanceNorm2d, Module, ModuleList, Parameter, ParameterList, ReLU
from torch.nn.functional import conv2d, pad


class WTDepthwiseConv2D(Module):
    """
    Depthwise convolution with additional multi-level wavelet branches (WTConv,
    "Wavelet Convolutions for Large Receptive Fields", Finder et al., ECCV 2024).
    Drop-in replacement for the depthwise conv of DepthSepConv2D:
    - `weight`/`bias` keep the exact names and shapes of the Conv2d they replace,
      so pretrained checkpoints load into the base path without remapping;
    - the wavelet branches are gated by zero-initialized per-channel scales, so at
      initialization the module computes exactly the same function as the
      pretrained depthwise conv (the gates move away from zero during training).
    """

    def __init__(self, channels, kernel_size=(3, 3), padding=(1, 1), wt_levels=2, wt_type="db1"):
        super(WTDepthwiseConv2D, self).__init__()
        # deferred import so the baseline code keeps working without PyWavelets
        from basic.wavelet_util import create_2d_wavelet_filter

        self.channels = channels
        self.conv_padding = padding
        self.wt_levels = wt_levels

        self.weight = Parameter(torch.empty(channels, 1, *kernel_size))
        self.bias = Parameter(torch.zeros(channels))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        wt_filter, iwt_filter = create_2d_wavelet_filter(wt_type, channels, channels)
        self.register_buffer("wt_filter", wt_filter)
        self.register_buffer("iwt_filter", iwt_filter)

        self.wavelet_convs = ModuleList([
            Conv2d(channels * 4, channels * 4, kernel_size, padding="same", groups=channels * 4, bias=False)
            for _ in range(wt_levels)
        ])
        self.wavelet_scale = ParameterList([
            Parameter(torch.zeros(1, channels * 4, 1, 1))
            for _ in range(wt_levels)
        ])

    def forward(self, x):
        from basic.wavelet_util import wavelet_2d_transform, inverse_2d_wavelet_transform

        base = conv2d(x, self.weight, self.bias, padding=self.conv_padding, groups=self.channels)

        # wavelet cascade: decompose, convolve the 4 sub-bands of each level
        # (gated by the zero-init scale), then recompose coarse-to-fine
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        curr_x_ll = x
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_x_ll = pad(curr_x_ll, (0, curr_shape[3] % 2, 0, curr_shape[2] % 2))

            curr_x = wavelet_2d_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0]

            b, c, _, h, w = curr_x.shape
            curr_x_tag = self.wavelet_scale[i] * self.wavelet_convs[i](curr_x.reshape(b, c * 4, h, w))
            curr_x_tag = curr_x_tag.reshape(b, c, 4, h, w)

            x_ll_in_levels.append(curr_x_tag[:, :, 0])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4])

        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop() + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), x_h_in_levels.pop()], dim=2)
            next_x_ll = inverse_2d_wavelet_transform(curr_x, self.iwt_filter)
            curr_shape = shapes_in_levels.pop()
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        return base + next_x_ll


class DepthSepConv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=None, padding=True, stride=(1, 1), dilation=(1, 1), wt_levels=0):
        super(DepthSepConv2D, self).__init__()

        self.padding = None

        if padding:
            if padding is True:
                padding = [int((k - 1) / 2) for k in kernel_size]
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.padding = [padding_h//2, padding_h-padding_h//2, padding_w//2, padding_w-padding_w//2]
                    padding = (0, 0)

        else:
            padding = (0, 0)
        if wt_levels > 0:
            # wavelet branches only support the plain depthwise case
            assert tuple(stride) == (1, 1) and tuple(dilation) == (1, 1) and self.padding is None
            self.depth_conv = WTDepthwiseConv2D(in_channels, kernel_size=kernel_size, padding=tuple(padding), wt_levels=wt_levels)
        else:
            self.depth_conv = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, groups=in_channels)
        self.point_conv = Conv2d(in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=(1, 1))
        self.activation = activation

    def forward(self, x):
        x = self.depth_conv(x)
        if self.padding:
            x = pad(x, self.padding)
        if self.activation:
            x = self.activation(x)
        x = self.point_conv(x)
        return x


class MixDropout(Module):
    def __init__(self, dropout_proba=0.4, dropout2d_proba=0.2):
        super(MixDropout, self).__init__()

        self.dropout = Dropout(dropout_proba)
        self.dropout2d = Dropout2d(dropout2d_proba)

    def forward(self, x):
        if random.random() < 0.5:
            return self.dropout(x)
        return self.dropout2d(x)

class ConvBlock(Module):

    def __init__(self, in_, out_, stride=(1, 1), k=3, activation=ReLU, dropout=0.4):
        super(ConvBlock, self).__init__()

        self.activation = activation()
        self.conv1 = Conv2d(in_channels=in_, out_channels=out_, kernel_size=k, padding=k // 2)
        self.conv2 = Conv2d(in_channels=out_, out_channels=out_, kernel_size=k, padding=k // 2)
        self.conv3 = Conv2d(out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = InstanceNorm2d(out_, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout / 2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)
        x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        return x

class DSCBlock(Module):

    def __init__(self, in_, out_, stride=(2, 1), activation=ReLU, dropout=0.4, wt_levels=0):
        super(DSCBlock, self).__init__()

        self.activation = activation()
        self.conv1 = DepthSepConv2D(in_, out_, kernel_size=(3, 3), wt_levels=wt_levels)
        self.conv2 = DepthSepConv2D(out_, out_, kernel_size=(3, 3), wt_levels=wt_levels)
        self.conv3 = DepthSepConv2D(out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=stride,
                                    wt_levels=wt_levels if tuple(stride) == (1, 1) else 0)
        self.norm_layer = InstanceNorm2d(out_, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout/2)

    def forward(self, x1):
        pos = random.randint(1, 3)
        x = self.conv1(x1)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)

        if pos == 3:
            x = self.dropout(x)
        x = x + x1 if x.size() == x1.size() else x
        return x