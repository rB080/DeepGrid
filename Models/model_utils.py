import torch
import torch.nn as nn



class SE_Block(nn.Module):
    def __init__(self, c, r=16):  # c -> no of channels; r-> reduction ratio
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, out_channels=1):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        X = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return X * self.sigmoid(x)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()


        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=True)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=False)


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Node(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, padding=0, mode='down'):
        super(Node, self).__init__()
        self.layer = BasicConv2d(in_channels, in_channels, kernel_size, 1, padding, dilation)

        self.md = mode
        if mode == 'down':
            self.mod = nn.MaxPool2d(2)
            self.out = BasicConv2d(in_channels, out_channels, 3, 1, 1, 1)
            self.sq = SE_Block(in_channels)
        else:
            self.mod = nn.ConvTranspose2d(in_channels *2 , in_channels, kernel_size=2, stride=2)
            self.out = BasicConv2d(in_channels, out_channels, 3, 1, 1, 1)
            self.sa = SpatialAttention()
            self.sq = SE_Block(in_channels)

    def forward(self, x, y=None, z=None):
        if y is not None: x = x + y
        if self.md == 'down':
            x = self.layer(x)
            xc = self.sq(x)
            xd = self.mod(xc)
            xd = self.out(xd)
            return x, xd, xc
        else:
            x = self.layer(x)
            if z is not None:
                z = self.sa(z)
                xu = torch.cat((x ,z) ,1)
            xu = self.mod(xu)
            xu = self.sq(xu)
            xu = self.out(xu)
            return x, xu


class Encoder_3(nn.Module):
    def __init__(self):
        super(Encoder_3, self).__init__()
        self.stream1 = Node(64, 128, 9, 1, 4, 'down')
        self.stream2 = Node(128, 256, 7, 1, 3, 'down')
        self.stream3 = Node(256, 512, 5, 1, 2, 'down')

    def forward(self, x1, x2, x3):
        x1, x1d, x1c = self.stream1(x1)
        x2, x2d, x2c = self.stream2(x2, x1d)
        x3, x3d, x3c = self.stream3(x3, x2d)

        return x1, x2, x3, x1c, x2c, x3c

class Encoder_4(nn.Module):
    def __init__(self):
        super(Encoder_4, self).__init__()
        self.stream1 = Node(64, 128, 9, 1, 4, 'down')
        self.stream2 = Node(128, 256, 7, 1, 3, 'down')
        self.stream3 = Node(256, 512, 5, 1, 2, 'down')
        self.stream4 = Node(512, 1024, 3, 1, 1, 'down')

    def forward(self, x1, x2, x3, x4):
        x1, x1d, x1c = self.stream1(x1)
        x2, x2d, x2c = self.stream2(x2, x1d)
        x3, x3d, x3c = self.stream3(x3, x2d)
        x4, x4d, x4c = self.stream4(x4, x3d)

        return x1, x2, x3, x4, x1c, x2c, x3c, x4c

class Encoder_5(nn.Module):
    def __init__(self):
        super(Encoder_5, self).__init__()
        self.stream1 = Node(64, 128, 9, 1, 4, 'down')
        self.stream2 = Node(128, 256, 7, 1, 3, 'down')
        self.stream3 = Node(256, 512, 5, 1, 2, 'down')
        self.stream4 = Node(512, 1024, 3, 1, 1, 'down')
        self.stream5 = Node(1024, 2048, 1, 1, 0, 'down')

    def forward(self, x1, x2, x3, x4, x5):
        x1, x1d, x1c = self.stream1(x1)
        x2, x2d, x2c = self.stream2(x2, x1d)
        x3, x3d, x3c = self.stream3(x3, x2d)
        x4, x4d, x4c = self.stream4(x4, x3d)
        x5, x5d, x5c = self.stream5(x5, x4d)

        return x1, x2, x3, x4, x5, x1c, x2c, x3c, x4c, x5c

class Decoder_3(nn.Module):
    def __init__(self):
        super(Decoder_3, self).__init__()
        self.stream1 = Node(64, 32, 9, 1, 4, 'up')
        self.stream2 = Node(128, 64, 7, 1, 3, 'up')
        self.stream3 = Node(256, 128, 5, 1, 2, 'up')

    def forward(self, x1, x2, x3, x1c, x2c, x3c):
        x3, x3u = self.stream3(x3, None, x3c)
        x2, x2u = self.stream2(x2, x3u, x2c)
        x1, x1u = self.stream1(x1, x2u, x1c)

        return x1, x2, x3

class Decoder_4(nn.Module):
    def __init__(self):
        super(Decoder_4, self).__init__()
        self.stream1 = Node(64, 32, 9, 1, 4, 'up')
        self.stream2 = Node(128, 64, 7, 1, 3, 'up')
        self.stream3 = Node(256, 128, 5, 1, 2, 'up')
        self.stream4 = Node(512, 256, 3, 1, 1, 'up')

    def forward(self, x1, x2, x3, x4, x1c, x2c, x3c, x4c):
        x4, x4u = self.stream4(x4, None, x4c)
        x3, x3u = self.stream3(x3, x4u, x3c)
        x2, x2u = self.stream2(x2, x3u, x2c)
        x1, x1u = self.stream1(x1, x2u, x1c)

        return x1, x2, x3, x4

class Decoder_5(nn.Module):
    def __init__(self):
        super(Decoder_5, self).__init__()
        self.stream1 = Node(64, 32, 9, 1, 4, 'up')
        self.stream2 = Node(128, 64, 7, 1, 3, 'up')
        self.stream3 = Node(256, 128, 5, 1, 2, 'up')
        self.stream4 = Node(512, 256, 3, 1, 1, 'up')
        self.stream5 = Node(1024, 512, 1, 1, 0, 'up')

    def forward(self, x1, x2, x3, x4, x5, x1c, x2c, x3c, x4c, x5c):
        x5, x5u = self.stream5(x5, None, x5c)
        x4, x4u = self.stream4(x4, x5u, x4c)
        x3, x3u = self.stream3(x3, x4u, x3c)
        x2, x2u = self.stream2(x2, x3u, x2c)
        x1, x1u = self.stream1(x1, x2u, x1c)

        return x1, x2, x3, x4, x5

class Midlayer_3(nn.Module):
    def __init__(self):
        super(Midlayer_3, self).__init__()
        self.stream1 = Node(64, 128, 7, 2, 6, 'down')
        self.stream2 = Node(128, 256, 7, 1, 3, 'down')
        self.stream3 = Node(256, 512, 3, 2, 2, 'down')

    def forward(self, x1, x2, x3):
        b, a, x1 = self.stream1(x1)
        b, a, x2 = self.stream2(x2)
        b, a, x3 = self.stream3(x3)
        return x1, x2, x3

class Midlayer_4(nn.Module):
    def __init__(self):
        super(Midlayer_4, self).__init__()
        self.stream1 = Node(64, 128, 7, 2, 6, 'down')
        self.stream2 = Node(128, 256, 7, 1, 3, 'down')
        self.stream3 = Node(256, 512, 3, 2, 2, 'down')
        self.stream4 = Node(512, 1024, 3, 1, 1, 'down')

    def forward(self, x1, x2, x3, x4):
        b, a, x1 = self.stream1(x1)
        b, a, x2 = self.stream2(x2)
        b, a, x3 = self.stream3(x3)
        b, a, x4 = self.stream4(x4)
        return x1, x2, x3, x4

class Midlayer_5(nn.Module):
    def __init__(self):
        super(Midlayer_5, self).__init__()
        self.stream1 = Node(64, 128, 7, 2, 6, 'down')
        self.stream2 = Node(128, 256, 7, 1, 3, 'down')
        self.stream3 = Node(256, 512, 3, 2, 2, 'down')
        self.stream4 = Node(512, 1024, 3, 1, 1, 'down')
        self.stream5 = Node(1024, 2048, 1, 1, 0, 'down')

    def forward(self, x1, x2, x3, x4, x5):
        b, a, x1 = self.stream1(x1)
        b, a, x2 = self.stream2(x2)
        b, a, x3 = self.stream3(x3)
        b, a, x4 = self.stream4(x4)
        b, a, x5 = self.stream5(x5)
        return x1, x2, x3, x4, x5