import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1dSamePadding(nn.Conv1d):
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class ResUnit(nn.Module):
    def __init__(self, c_in, c_out, k=8, dilation=1, stride=1, bias=True):
        super().__init__()

        self.layers = nn.Sequential(Conv1dSamePadding(in_channels=c_in, out_channels=c_out,
                                                      kernel_size=k, dilation=dilation, stride=stride, bias=bias),
                                    nn.GELU(),
                                    nn.BatchNorm1d(c_out)
                                    )
        if c_in > 1 and c_in != c_out:
            self.match_residual = True
            self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=1)
        else:
            self.match_residual = False

    def forward(self, x):
        if self.match_residual:
            x_bottleneck = self.conv(x)
            x = self.layers(x)

            return torch.add(x_bottleneck, x)
        else:
            return torch.add(x, self.layers(x))


class DilatedBlock(nn.Module):
    def __init__(self, c_in=32, c_out=32,
                 kernel_size=8, dilation_list=[1, 2, 4, 8],
                 bias=True):
        super().__init__()

        layers = []
        for i, dilation in enumerate(dilation_list):
            if i == 0:
                layers.append(ResUnit(c_in, c_out, k=kernel_size, dilation=dilation, bias=bias))
            else:
                layers.append(ResUnit(c_out, c_out, k=kernel_size, dilation=dilation, bias=bias))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return x
