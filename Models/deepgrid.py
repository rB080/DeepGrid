from Models.model_utils import *
import torch.nn.functional as F
import torch.nn as nn


class Grid_5(nn.Module):
    def __init__(self, layers=3):
        super(Grid_5, self).__init__()
        self.down = nn.MaxPool2d(2)
        self.in1 = nn.Conv2d(3, 64, 1)
        self.in2 = nn.Conv2d(64, 128, 1)
        self.in3 = nn.Conv2d(128, 256, 1)
        self.in4 = nn.Conv2d(256, 512, 1)
        self.in5 = nn.Conv2d(512, 1024, 1)

        self.e1 = Encoder_5()
        self.e2 = Encoder_5()
        if layers >= 3: self.e3 = Encoder_5()
        if layers >= 4: self.e4 = Encoder_5()

        self.ml = Midlayer_5()

        if layers >= 4: self.d4 = Decoder_5()
        if layers >= 3: self.d3 = Decoder_5()
        self.d2 = Decoder_5()
        self.d1 = Decoder_5()

        self.out = nn.Conv2d(64, 1, 1)
        self.layers = layers

    def forward(self, x):
        x1 = self.in1(x)
        x2 = self.in2(self.down(x1))
        x3 = self.in3(self.down(x2))
        x4 = self.in4(self.down(x3))
        x5 = self.in5(self.down(x4))

        x1, x2, x3, x4, x5, xe11, xe12, xe13, xe14, xe15 = self.e1(x1, x2, x3, x4, x5)
        x1, x2, x3, x4, x5, xe21, xe22, xe23, xe24, xe25 = self.e2(x1, x2, x3, x4, x5)
        if self.layers >= 3: x1, x2, x3, x4, x5, xe31, xe32, xe33, xe34, xe35 = self.e3(x1, x2, x3, x4, x5)
        if self.layers >= 4: x1, x2, x3, x4, x5, xe41, xe42, xe43, xe44, xe45 = self.e4(x1, x2, x3, x4, x5)

        x1, x2, x3, x4, x5 = self.ml(x1, x2, x3, x4, x5)

        if self.layers >= 4: x1, x2, x3, x4, x5 = self.d4(x1, x2, x3, x4, x5, xe41, xe42, xe43, xe44, xe45)
        if self.layers >= 3: x1, x2, x3, x4, x5 = self.d3(x1, x2, x3, x4, x5, xe31, xe32, xe33, xe34, xe35)
        x1, x2, x3, x4, x5 = self.d2(x1, x2, x3, x4, x5, xe21, xe22, xe23, xe24, xe25)
        x1, x2, x3, x4, x5 = self.d1(x1, x2, x3, x4, x5, xe11, xe12, xe13, xe14, xe15)
        x1 = self.out(x1)
        return x1


class Grid_4(nn.Module):
    def __init__(self, layers=3):
        super(Grid_4, self).__init__()
        self.down = nn.MaxPool2d(2)
        self.in1 = nn.Conv2d(3, 64, 1)
        self.in2 = nn.Conv2d(64, 128, 1)
        self.in3 = nn.Conv2d(128, 256, 1)
        self.in4 = nn.Conv2d(256, 512, 1)

        self.e1 = Encoder_4()
        self.e2 = Encoder_4()
        if layers >= 3: self.e3 = Encoder_4()
        if layers >= 4: self.e4 = Encoder_4()

        self.ml = Midlayer_4()

        if layers >= 4: self.d4 = Decoder_4()
        if layers >= 3: self.d3 = Decoder_4()
        self.d2 = Decoder_4()
        self.d1 = Decoder_4()

        self.out = nn.Conv2d(64, 1, 1)
        self.layers = layers

    def forward(self, x):
        x1 = self.in1(x)
        x2 = self.in2(self.down(x1))
        x3 = self.in3(self.down(x2))
        x4 = self.in4(self.down(x3))

        x1, x2, x3, x4, xe11, xe12, xe13, xe14 = self.e1(x1, x2, x3, x4)
        x1, x2, x3, x4, xe21, xe22, xe23, xe24 = self.e2(x1, x2, x3, x4)
        if self.layers >= 3: x1, x2, x3, x4, xe31, xe32, xe33, xe34 = self.e3(x1, x2, x3, x4)
        if self.layers >= 4: x1, x2, x3, x4, xe41, xe42, xe43, xe44 = self.e4(x1, x2, x3, x4)

        x1, x2, x3, x4 = self.ml(x1, x2, x3, x4)

        if self.layers >= 4: x1, x2, x3, x4 = self.d4(x1, x2, x3, x4, xe41, xe42, xe43, xe44)
        if self.layers >= 3: x1, x2, x3, x4 = self.d3(x1, x2, x3, x4, xe31, xe32, xe33, xe34)
        x1, x2, x3, x4 = self.d2(x1, x2, x3, x4, xe21, xe22, xe23, xe24)
        x1, x2, x3, x4 = self.d1(x1, x2, x3, x4, xe11, xe12, xe13, xe14)
        x1 = self.out(x1)
        return x1


class Grid_3(nn.Module):
    def __init__(self, layers=3):
        super(Grid_3, self).__init__()
        self.down = nn.MaxPool2d(2)
        self.in1 = nn.Conv2d(3, 64, 1)
        self.in2 = nn.Conv2d(64, 128, 1)
        self.in3 = nn.Conv2d(128, 256, 1)

        self.e1 = Encoder_3()
        self.e2 = Encoder_3()
        if layers >= 3: self.e3 = Encoder_3()
        if layers >= 4: self.e4 = Encoder_3()

        self.ml = Midlayer_3()

        if layers >= 3: self.d4 = Decoder_3()
        if layers >= 4: self.d3 = Decoder_3()
        self.d2 = Decoder_3()
        self.d1 = Decoder_3()

        self.out = nn.Conv2d(64, 1, 1)
        self.layers = layers

    def forward(self, x):
        x1 = self.in1(x)
        x2 = self.in2(self.down(x1))
        x3 = self.in3(self.down(x2))

        x1, x2, x3, xe11, xe12, xe13 = self.e1(x1, x2, x3)
        x1, x2, x3, xe21, xe22, xe23 = self.e2(x1, x2, x3)
        if self.layers >= 3: x1, x2, x3, xe31, xe32, xe33 = self.e3(x1, x2, x3)
        if self.layers >= 4: x1, x2, x3, xe41, xe42, xe43 = self.e4(x1, x2, x3)

        x1, x2, x3 = self.ml(x1, x2, x3)

        if self.layers >= 4: x1, x2, x3 = self.d4(x1, x2, x3, xe41, xe42, xe43)
        if self.layers >= 3: x1, x2, x3 = self.d3(x1, x2, x3, xe31, xe32, xe33)
        x1, x2, x3 = self.d2(x1, x2, x3, xe21, xe22, xe23)
        x1, x2, x3 = self.d1(x1, x2, x3, xe11, xe12, xe13)
        x1 = self.out(x1)
        return x1

def give_model(device, streams=4, columns=3, load_path=None):
    assert streams in [3,4,5], 'Unavailable stream number!'
    assert columns in [2, 3, 4], 'Unavailable column number!'
    Grids = [Grid_3, Grid_4, Grid_5]
    model = Grids[streams - 3](layers=columns).to(device)
    if load_path is not None: model.load_state_dict(torch.load(load_path, map_location=device))
    return model