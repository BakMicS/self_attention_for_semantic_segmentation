import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from model.san import Subtraction, Subtraction2, Aggregation, position

__all__ = ["DeepLabV3_SAN"]


class DeepLabV3_SAN(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from:
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    + Option to add SA-blocks from:
    "Exploring Self-Attention for Image Recognition"
    <https://arxiv.org/abs/2004.13621>

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class DeepLabHead_SAN(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead_SAN, self).__init__(
            ASPP_SAN(in_channels, [6, 12, 18], self_attention=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, kernel_size=5):  # Original ASPP kernel size = 3
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(dilation * (kernel_size - 1) + 1) // 2, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPConv_SAN(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            Bottleneck(1, in_channels, in_channels // 16, in_channels // 4, out_channels, share_planes=8, kernel_size=7, stride=1, dialation=dilation),
            nn.BatchNorm2d(out_channels),  # TODO: add if output conv is added in Bottleneck
            nn.ReLU()
        ]
        super(ASPPConv_SAN, self).__init__(*modules)


class SAM(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, out_planes, share_planes, kernel_size=3, stride=1, dilation=1):
        super(SAM, self).__init__()
        self.sa_type, self.kernel_size, self.stride = sa_type, kernel_size, stride
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        if sa_type == 0:  # pairwise
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes + 2), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes + 2, rel_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(rel_planes), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes, out_planes // share_planes, kernel_size=1))
            self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
            self.subtraction = Subtraction(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.subtraction2 = Subtraction2(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.softmax = nn.Softmax(dim=-2)
        else:  # patchwise
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes * (pow(kernel_size, 2) + 1)), nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes * (pow(kernel_size, 2) + 1), out_planes // share_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_planes // share_planes), nn.ReLU(inplace=True),
                                        nn.Conv2d(out_planes // share_planes, pow(kernel_size, 2) * out_planes // share_planes, kernel_size=1))
            self.unfold_i = nn.Unfold(kernel_size=1, dilation=dilation, padding=0, stride=stride)
            self.unfold_j = nn.Unfold(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)

    def forward(self, x):
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        if self.sa_type == 0:  # pairwise
            p = self.conv_p(position(x.shape[2], x.shape[3], x.is_cuda))
            w = self.softmax(self.conv_w(
                torch.cat([self.subtraction2(x1, x2), self.subtraction(p).repeat(x.shape[0], 1, 1, 1)], 1)))
        else:  # patchwise
            if self.stride != 1:
                x1 = self.unfold_i(x1)
            x1 = x1.view(x.shape[0], -1, 1, x.shape[2] * x.shape[3])
            x2 = self.unfold_j(self.pad(x2)).view(x.shape[0], -1, 1, x1.shape[-1])
            w = self.conv_w(torch.cat([x1, x2], 1)).view(x.shape[0], -1, pow(self.kernel_size, 2), x1.shape[-1])
        x = self.aggregation(x3, w)
        return x


class Bottleneck(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, kernel_size=7, stride=1, dialation=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.sam = SAM(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride, dialation)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # out = self.relu(self.bn1(x))  # removed
        out = self.relu(self.bn2(self.sam(x)))
        out = self.conv(out)
        return out


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP_SAN(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256, self_attention=True):
        super(ASPP_SAN, self).__init__()

        modules = []

        if self_attention:
            modules.append(ASPPConv_SAN(in_channels, out_channels, 1))
        else:  # original convolution ASPP
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            if self_attention:
                modules.append(ASPPConv_SAN(in_channels, out_channels, rate))
            else:  # original convolution ASPP
                modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
