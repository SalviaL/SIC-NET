import torch.nn as nn
import torch
import torch.functional as F
from itertools import product


class ResNet_block(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernal_size=3,
                 padding=1):
        super(ResNet_block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, 1, padding=1),
            nn.BatchNorm2d(input_channels), nn.Tanh())
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 1, padding=1),
            nn.BatchNorm2d(output_channels), nn.Tanh())
        self.bn = nn.Sequential(nn.BatchNorm2d(output_channels), nn.Tanh())

    def forward(self, x):
        input_ = x
        x = self.conv1(x)
        x = self.conv2(x)
        # input_ = self.conv3(input_)
        return self.bn(x + input_)


class ReNet_layer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ReNet_layer, self).__init__()
        if output_channels % 2 != 0:
            output_channels += 1
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.Vlayer = nn.GRU(input_channels,
                             int(input_channels / 2),
                             1,
                             bidirectional=True)
        self.Hlayer = nn.GRU(input_channels,
                             int(output_channels / 2),
                             1,
                             bidirectional=True)

    def forward(self, x):
        b, c, m, n = x.size()
        x = x.permute(2, 3, 0, 1)
        V_map = torch.zeros(m, n, b, self.input_channels).cuda()
        H_map = torch.zeros(m, n, b, self.output_channels).cuda()
        for i in range(0, n):
            V_map[:, i, :, :], _ = self.Vlayer(x[:, i, :, :])
        for i in range(0, m):
            H_map[i, :, :, :], _ = self.Hlayer(V_map[i, :, :, :])
        return H_map.permute(2, 3, 0, 1)


class Re_CNN_block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Re_CNN_block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, 1, padding=1),
            nn.BatchNorm2d(input_channels), nn.ReLU())

        self.renet1 = ReNet_layer(input_channels, input_channels)

        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 1, padding=1),
            nn.BatchNorm2d(output_channels), nn.ReLU())

        self.renet2 = ReNet_layer(input_channels, output_channels)

        self.bn = nn.Sequential(nn.BatchNorm2d(output_channels), nn.ReLU())

    def forward(self, x):
        input_ = x
        x1 = self.conv1(x)
        x1 = self.renet2(x1)
        x2 = self.renet1(x)
        x2 = self.conv2(x2)
        return self.bn(x1 + x2 + input_)


class model_sic_net(nn.Module):
    def __init__(self,
                 input_feature_map_channnels=1024,
                 num_classes=7,
                 num_CTU=3,
                 num_FCN_layers=3):
        '''
        input_feature_map_channnels: the dimension of encoding result\n
        num_classes: the total number of classes\n
        num_CTU: the number of crossing transfer unit\n
        num_FCN_layers: number of finnal FCN layers (classifing layer included)\n
        ![]()
        '''
        super(model_sic_net, self).__init__()
        self.num_classes = num_classes
        self.conv1x1 = nn.Sequential(
            nn.BatchNorm2d(input_feature_map_channnels), nn.Tanh(),
            nn.Conv2d(input_feature_map_channnels, 512, 1),
            nn.BatchNorm2d(512), nn.Tanh())
        self.CTU = []
        for i in range(num_CTU):
            self.CTU += [Re_CNN_block(512, 512)]
        self.CTU = nn.Sequential(*self.CTU)
        self.bn = nn.Sequential(nn.BatchNorm2d(512), nn.Tanh())

        self.fcn = []
        for i in range(num_FCN_layers - 1):
            self.fcn += [
                nn.Sequential(nn.Dropout2d(), nn.Conv2d(512, 512, 3, 1, 1),
                              nn.BatchNorm2d(512), nn.Tanh())
            ]
        self.fcn = nn.Sequential(*self.fcn)

        self.out = nn.Conv2d(512, num_classes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1x1(x)
        input_ = x
        x = self.CTU(x)
        x = self.bn(x + input_)
        x = self.fcn(x)
        return self.out(x)


if __name__ == "__main__":
    # select_a_model(13)
    a = torch.rand(4, 512, 32, 32).cuda()
    model = model_sic_net(512, 7, 3, 3).cuda()
    c = model(a)
    print(c.size())
