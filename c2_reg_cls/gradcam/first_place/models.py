import torch
from torch import nn

import timm

from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

from . import timm4smp
from .timm4smp.models.layers.conv2d_same import Conv2dSame
from .covn3dsame import Conv3dSame

n_slice_per_c = 15
n_blocks = 4
_in_chans = 6

def convert_3d(module):
    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = torch.nn.BatchNorm3d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    elif isinstance(module, Conv2dSame):
        module_output = Conv3dSame(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0]))

    elif isinstance(module, torch.nn.Conv2d):
        module_output = torch.nn.Conv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0]))

    elif isinstance(module, torch.nn.MaxPool2d):
        module_output = torch.nn.MaxPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        module_output = torch.nn.AvgPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
        )

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_3d(child)
        )
    del module

    return module_output


class TimmSegModel(nn.Module):
    def __init__(self, backbone, segtype='unet', pretrained=False):
        super(TimmSegModel, self).__init__()

        self.encoder = timm4smp.create_model(
            backbone,
            in_chans=3,
            features_only=True,
            pretrained=pretrained
        )
        g = self.encoder(torch.rand(1, 3, 64, 64))
        encoder_channels = [1] + [_.shape[1] for _ in g]
        decoder_channels = [256, 128, 64, 32, 16]
        if segtype == 'unet':
            self.decoder = UnetDecoder(
                encoder_channels=encoder_channels[:n_blocks + 1],
                decoder_channels=decoder_channels[:n_blocks],
                n_blocks=n_blocks,
            )
        self.segmentation_head = nn.Conv2d(decoder_channels[n_blocks - 1], 7, kernel_size=(3, 3), stride=(1, 1),
                                           padding=(1, 1))

    def forward(self, x):
        global_features = [0] + self.encoder(x)[:n_blocks]
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features


class TimmModel(nn.Module):
    def __init__(self, backbone, image_size, pretrained=False):
        super(TimmModel, self).__init__()
        self.image_size = image_size
        self.encoder = timm.create_model(
            backbone,
            in_chans=_in_chans,
            num_classes=1,
            features_only=False,
            drop_rate=0,
            drop_path_rate=0,
            pretrained=pretrained
        )

        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone or 'nfnet' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()

        self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=0, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),
        )
        self.lstm2 = nn.LSTM(hdim, 256, num_layers=2, dropout=0, bidirectional=True, batch_first=True)
        self.head2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x):  # (bs, nc*7, ch, sz, sz)
        bs = x.shape[0]
        x = x.view(bs * n_slice_per_c * 7, _in_chans, self.image_size, self.image_size)
        feat = self.encoder(x)
        feat = feat.view(bs, n_slice_per_c * 7, -1)
        feat1, _ = self.lstm(feat)
        feat1 = feat1.contiguous().view(bs * n_slice_per_c * 7, 512)
        feat2, _ = self.lstm2(feat)

        return self.head(feat1), self.head2(feat2[:, 0])


class Timm1BoneModel(nn.Module):
    def __init__(self, backbone, image_size, pretrained=False):
        super(Timm1BoneModel, self).__init__()
        self.image_size = image_size

        self.encoder = timm.create_model(
            backbone,
            in_chans=_in_chans,
            num_classes=1,
            features_only=False,
            drop_rate=0,
            drop_path_rate=0,
            pretrained=pretrained
        )

        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone or 'nfnet' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()

        self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=0, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        bs = x.shape[0]
        x = x.view(bs * n_slice_per_c, _in_chans, self.image_size, self.image_size)
        feat = self.encoder(x)
        feat = feat.view(bs, n_slice_per_c, -1)
        feat, _ = self.lstm(feat)
        feat = feat.contiguous().view(bs * n_slice_per_c, -1)
        feat = self.head(feat)
        feat = feat.view(bs, n_slice_per_c).contiguous()

        return feat
