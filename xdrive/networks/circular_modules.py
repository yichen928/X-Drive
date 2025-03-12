# https://github.com/kazuto1011/circular-conv-pytorch
import math
from collections import OrderedDict

import collections
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.autograd import Function

class CircularConv2d3x3(nn.Module):
    def __init__(self, n_in, n_out, **kwargs):
        super(CircularConv2d3x3, self).__init__()
        self.network = nn.Sequential(
            nn.ReflectionPad2d(padding=(1, 1, 0, 0)),
            nn.ZeroPad2d(padding=(0, 0, 1, 1)),
            nn.Conv2d(n_in, n_out, kernel_size=3, padding=0, **kwargs),
        )

    def forward(self, x):
        output = self.network(x)
        return output


class CircularConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        if 'padding' in kwargs:
            self.is_pad = True
            if isinstance(kwargs['padding'], int):
                h1 = h2 = v1 = v2 = kwargs['padding']
            elif isinstance(kwargs['padding'], tuple):
                if len(kwargs['padding']) == 4:
                    # h1, h2, v1, v2 = kwargs['padding']
                    v1, v2, h1, h2 = kwargs['padding']
                else:
                    assert len(kwargs['padding']) == 2
                    v1 = v2 = kwargs['padding'][0]
                    h1 = h2 = kwargs['padding'][1]
            else:
                raise NotImplementedError
            self.h_pad, self.v_pad = (h1, h2, 0, 0), (0, 0, v1, v2)
            del kwargs['padding']
        else:
            self.is_pad = False

        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.is_pad:
            if sum(self.h_pad) > 0:
                x = nn.functional.pad(x, self.h_pad, mode="circular")  # horizontal pad
            if sum(self.v_pad) > 0:
                x = nn.functional.pad(x, self.v_pad, mode="constant")  # vertical pad
        x = self._conv_forward(x, self.weight, self.bias)
        return x
