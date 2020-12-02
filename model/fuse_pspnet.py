import torch
from torch import nn
import torch.nn.functional as F
import copy
from torch.autograd import Variable

import model.resnet as models
from model.pspnet import PPM


class FusedPSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(FusedPSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        # depth branch
        self.layer0_d = copy.deepcopy(self.layer0)
        self.layer1_d = copy.deepcopy(self.layer1)
        self.layer2_d = copy.deepcopy(self.layer2)
        self.layer3_d = copy.deepcopy(self.layer3)
        self.layer4_d = copy.deepcopy(self.layer4)

        for layer in [self.layer3, self.layer3_d]:
            for n, m in layer.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        for layer in [self.layer4, self.layer4_d]:
            for n, m in layer.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )
            self.aux_d = copy.deepcopy(self.aux)

    def forward(self, x, x_d, y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        # depth branch
        x_d = self.layer0_d(x_d)
        x_d = self.layer1_d(x_d)
        x_d = self.layer2_d(x_d)
        x_d_tmp = self.layer3_d(x_d)
        x_d = self.layer4_d(x_d_tmp)
        # fuse
        x = torch.add(x, x_d)
        x = torch.div(x, 2)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            aux_d = self.aux_d(x_d_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
                aux_d = F.interpolate(aux_d, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            aux_d_loss = self.criterion(aux_d, y)
            return x.max(1)[1], main_loss, aux_loss, aux_d_loss
        else:
            return x


def channel_attention(num_channel):
    pool = nn.AdaptiveAvgPool2d(1)
    conv = nn.Conv2d(num_channel, num_channel, kernel_size=1)
    activation = nn.Sigmoid()
    return nn.Sequential(*[pool, conv, activation])


def make_agant_layer(inplanes, planes):
    layers = nn.Sequential(
        nn.Conv2d(inplanes, planes, kernel_size=1,
                  stride=1, padding=0, bias=False),
        nn.BatchNorm2d(planes),
        nn.ReLU(inplace=True)
    )
    return layers


class AttentionFusedPSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(AttentionFusedPSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        # depth branch
        self.layer0_d = copy.deepcopy(self.layer0)
        self.layer1_d = copy.deepcopy(self.layer1)
        self.layer2_d = copy.deepcopy(self.layer2)
        self.layer3_d = copy.deepcopy(self.layer3)
        self.layer4_d = copy.deepcopy(self.layer4)

        # attention
        self.attention_rgb = channel_attention(2048)
        self.attention_depth = channel_attention(2048)

        for layer in [self.layer3, self.layer3_d]:
            for n, m in layer.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        for layer in [self.layer4, self.layer4_d]:
            for n, m in layer.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )
            self.aux_d = copy.deepcopy(self.aux)

    def forward(self, x, x_d, y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        # depth branch
        x_d = self.layer0_d(x_d)
        x_d = self.layer1_d(x_d)
        x_d = self.layer2_d(x_d)
        x_d_tmp = self.layer3_d(x_d)
        x_d = self.layer4_d(x_d_tmp)
        # attention
        atten_rgb = self.attention_rgb(x)
        atten_depth = self.attention_depth(x_d)
        # fuse
        x = x.mul(atten_rgb) + x_d.mul(atten_depth)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            aux_d = self.aux_d(x_d_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
                aux_d = F.interpolate(aux_d, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            aux_d_loss = self.criterion(aux_d, y)
            return x.max(1)[1], main_loss, aux_loss, aux_d_loss
        else:
            return x


class Attention_v1_FusedPSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(Attention_v1_FusedPSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        # depth branch
        self.layer0_d = copy.deepcopy(self.layer0)
        self.layer1_d = copy.deepcopy(self.layer1)
        self.layer2_d = copy.deepcopy(self.layer2)
        self.layer3_d = copy.deepcopy(self.layer3)
        self.layer4_d = copy.deepcopy(self.layer4)

        # attention
        self.attention_rgb = channel_attention(2048)
        self.attention_depth = channel_attention(2048)
        self.agant = make_agant_layer(2048, 2048)

        for layer in [self.layer3, self.layer3_d]:
            for n, m in layer.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        for layer in [self.layer4, self.layer4_d]:
            for n, m in layer.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )
            self.aux_d = copy.deepcopy(self.aux)

    def forward(self, x, x_d, y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        # depth branch
        x_d = self.layer0_d(x_d)
        x_d = self.layer1_d(x_d)
        x_d = self.layer2_d(x_d)
        x_d_tmp = self.layer3_d(x_d)
        x_d = self.layer4_d(x_d_tmp)
        # attention
        atten_rgb = self.attention_rgb(x)
        atten_depth = self.attention_depth(x_d)
        # fuse
        x = x.mul(atten_rgb) + x_d.mul(atten_depth)
        x = self.agant(x)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            aux_d = self.aux_d(x_d_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
                aux_d = F.interpolate(aux_d, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            aux_d_loss = self.criterion(aux_d, y)
            return x.max(1)[1], main_loss, aux_loss, aux_d_loss
        else:
            return x


class Attention_v2_FusedPSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(Attention_v2_FusedPSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        # depth branch
        self.layer0_d = copy.deepcopy(self.layer0)
        self.layer1_d = copy.deepcopy(self.layer1)
        self.layer2_d = copy.deepcopy(self.layer2)
        self.layer3_d = copy.deepcopy(self.layer3)
        self.layer4_d = copy.deepcopy(self.layer4)

        # attention
        self.a_0 = channel_attention(128)
        self.a_0_d = channel_attention(128)
        self.a_1 = channel_attention(256)
        self.a_1_d = channel_attention(256)
        self.a_2 = channel_attention(512)
        self.a_2_d = channel_attention(512)
        self.a_3 = channel_attention(1024)
        self.a_3_d = channel_attention(1024)
        self.a_4 = channel_attention(2048)
        self.a_4_d = channel_attention(2048)
        self.agant = make_agant_layer(2048, 2048)

        for layer in [self.layer3, self.layer3_d]:
            for n, m in layer.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        for layer in [self.layer4, self.layer4_d]:
            for n, m in layer.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )
            self.aux_d = copy.deepcopy(self.aux)

    def forward(self, x, x_d, y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        # depth branch
        x_d_0 = self.layer0_d(x_d)
        x_d_1 = self.layer1_d(x_d_0)
        x_d_2 = self.layer2_d(x_d_1)
        x_d_tmp = self.layer3_d(x_d_2)
        x_d_4 = self.layer4_d(x_d_tmp)

        # rgb branch
        x = self.layer0(x)
        x = x.mul(self.a_0(x)) + x_d_0.mul(self.a_0_d(x_d_0))
        x = self.layer1(x)
        x = x.mul(self.a_1(x)) + x_d_1.mul(self.a_1_d(x_d_1))
        x = self.layer2(x)
        x = x.mul(self.a_2(x)) + x_d_2.mul(self.a_2_d(x_d_2))
        x_tmp = self.layer3(x)
        x_tmp = x_tmp.mul(self.a_3(x_tmp)) + x_d_tmp.mul(self.a_3_d(x_d_tmp))
        x = self.layer4(x_tmp)

        # fuse
        x = x.mul(self.a_4(x)) + x_d_4.mul(self.a_4_d(x_d_4))
        x = self.agant(x)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            aux_d = self.aux_d(x_d_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
                aux_d = F.interpolate(aux_d, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            aux_d_loss = self.criterion(aux_d, y)
            return x.max(1)[1], main_loss, aux_loss, aux_d_loss
        else:
            return x


def fuse(x, x_d):
    out = torch.add(x, x_d)
    out = torch.div(out, 2)
    return out


class DeepFusedPSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(DeepFusedPSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        # depth branch
        self.layer0_d = copy.deepcopy(self.layer0)
        self.layer1_d = copy.deepcopy(self.layer1)
        self.layer2_d = copy.deepcopy(self.layer2)
        self.layer3_d = copy.deepcopy(self.layer3)
        self.layer4_d = copy.deepcopy(self.layer4)

        for layer in [self.layer3, self.layer3_d]:
            for n, m in layer.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        for layer in [self.layer4, self.layer4_d]:
            for n, m in layer.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )
            self.aux_d = copy.deepcopy(self.aux)

    def forward(self, x, x_d, y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        # depth branch
        x_d_0 = self.layer0_d(x_d)
        x_d_1 = self.layer1_d(x_d_0)
        x_d_2 = self.layer2_d(x_d_1)
        x_d_3 = self.layer3_d(x_d_2)
        x_d_4 = self.layer4_d(x_d_3)

        # RGB branch
        x = self.layer0(x)
        x = self.layer1(fuse(x, x_d_0))
        x = self.layer2(fuse(x, x_d_1))
        x_tmp = self.layer3(fuse(x, x_d_2))
        x = self.layer4(fuse(x_tmp, x_d_3))
        # fuse
        x = fuse(x, x_d_4)

        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            aux_d = self.aux_d(x_d_3)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
                aux_d = F.interpolate(aux_d, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            aux_d_loss = self.criterion(aux_d, y)
            return x.max(1)[1], main_loss, aux_loss, aux_d_loss
        else:
            return x


class ShallowFusedPSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(ShallowFusedPSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        # depth branch
        self.layer0_d = copy.deepcopy(self.layer0)

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

        # placeholder
        self.constant = torch.Tensor([0])
        self.register_buffer('aux_loss_d', self.constant)

    def forward(self, x, x_d, y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        # depth branch
        x_d = self.layer0_d(x_d)

        x = self.layer0(x)
        # fuse
        x = torch.add(x, x_d)
        x = torch.div(x, 2)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)

        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss, Variable(self.aux_loss_d)
        else:
            return x