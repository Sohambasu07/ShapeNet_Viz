import torch
import torch.nn as nn

''' ResNet Block from https://github.com/kenshohara/3D-ResNets-PyTorch'''


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    if in_channels <= 32:
        num_groups = in_channels // 4
    else:
        num_groups = 32

    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, d, h, w = q.shape
        q = q.reshape(b, c, d * h * w)
        q = q.permute(0, 2, 1)  # b,dhw,c
        k = k.reshape(b, c, d * h * w)  # b,c,dhw
        w_ = torch.bmm(q, k)  # b,dhw,dhw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, d * h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, d, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Decoder3D(nn.Module):
    def __init__(self):
        self.conv_in = torch.nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.res1 = ResNetBlock(256, 256)
        self.attn1 = AttnBlock(256)
        self.res2 = ResNetBlock(256, 256)
        # torch.nn.Upsample(scale_factor=2)
        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=0)

        self.res3 = ResNetBlock(256, 128)
        self.attn2 = AttnBlock(128)
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=0)

        self.res4 = ResNetBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=0)
        self.res5 = ResNetBlock(64, 64)
        self.res6 = ResNetBlock(64, 64)
        self.norm = Normalize
        self.swish = nonlinearity

        self.conv_out = torch.nn.Conv3d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_in(x)

        x = self.res1(x)
        x = self.attn1(x)
        x = self.res2(x)
        x = self.up1(x)

        x = self.res3(x)
        x = self.attn2(x)
        x = self.up1(x)
        x = self.res4(x)
        x = self.up2(x)

        x = self.res5(x)
        x = self.res6(x)
        x = self.norm(x)
        x = self.swish(x)

        x = self.conv_out(x)

        return x
