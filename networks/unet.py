# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform

import torch.nn.functional as F
import cv2
class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])
        self.boundary_predictor = nn.Conv2d(self.ft_chns[4], 1, kernel_size=1)
    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        boundary_prediction = self.boundary_predictor(x4)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.representation = nn.Sequential(
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True), # nn.Dropout1d(0.1),
                nn.Linear(256,256),
            )
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
      
         

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        
        x11 = self.up1(x4, x3)
        x22 = self.up2(x11, x2)
        x33 = self.up3(x22, x1)
        x = self.up4(x33, x0)
        output = self.out_conv(x)
        b,c,h,w = x4.size()
        rep_out = self.representation(x4.permute(0,2,3,1).contiguous().view(b*h*w, c))
        rep = rep_out.view(b,h,w,256).permute(0,3,1,2)
        return output,x4,rep,[x11,x22,x33,x]
class Decoder_DS(nn.Module):
    def __init__(self, params):
        super(Decoder_DS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class Decoder_URDS(nn.Module):
    def __init__(self, params):
        super(Decoder_URDS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)
        self.feature_noise = FeatureNoise()

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        if self.training:
            dp3_out_seg = self.out_conv_dp3(Dropout(x, p=0.5))
        else:
            dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        if self.training:
            dp2_out_seg = self.out_conv_dp2(FeatureDropout(x))
        else:
            dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        if self.training:
            dp1_out_seg = self.out_conv_dp1(self.feature_noise(x))
        else:
            dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


def Dropout(x, p=0.5):
    x = torch.nn.functional.dropout2d(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output,boundary,feature, = self.decoder(feature)
        return output,boundary,feature


class UNet_DS(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_DS, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder_DS(params)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg = self.decoder(
            feature, shape)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class UNet_CCT(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_CCT, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [Dropout(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        return main_seg, aux_seg1


class UNet_CCT_3H(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_CCT_3H, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)
        self.aux_decoder2 = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [Dropout(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        aux2_feature = [FeatureNoise()(i) for i in feature]
        aux_seg2 = self.aux_decoder1(aux2_feature)
        return main_seg, aux_seg1, aux_seg2


class Edge_Module(nn.Module):
    def __init__(self, in_channels=[16,32,64,128], mid_channels=64):
        super(Edge_Module, self).__init__()
        self.conv1 = nn.Conv2d(in_channels[3], mid_channels, 1)
        self.conv2 = nn.Conv2d(in_channels[2], mid_channels, 1)
        self.conv3 = nn.Conv2d(in_channels[1], mid_channels, 1)
        self.conv4 = nn.Conv2d(in_channels[0], mid_channels, 1)
        self.conv5_2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.BEM = BoundaryEnhancementModule(in_chs=1, out_chs=256)
        self.classifier = nn.Conv2d(768, 1, kernel_size=3, padding=1)
        self.classifier2 = nn.Conv2d(512, 1, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, feature,x,da0,da1):
        x1 = feature[0]
        x2 = feature[1]
        x3 = feature[2]
        x4 = feature[3]
        bem_out = self.BEM(x)
        b,c,h,w = x4.size()
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv5_2(x1))
        x2 = F.relu(self.conv2(x2))
        x2 = F.relu(self.conv5_2(x2))
        x3 = F.relu(self.conv3(x3))
        x3 = F.relu(self.conv5_2(x3))
        x4 = F.relu(self.conv4(x4))
        x4 = F.relu(self.conv5_2(x4))
       
        
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=(h, w), mode='bilinear', align_corners=True)
        x1 = F.interpolate(x1, size=(h, w), mode='bilinear', align_corners=True)
       
        da0 = F.interpolate(da0, size=(h, w), mode='bilinear', align_corners=True)
        da1 = F.interpolate(da1, size=(h, w), mode='bilinear', align_corners=True)
        #x = torch.cat([x1, x2, x3, x4, bem_out,da0,da1], dim=1)
        x2 = torch.cat([x1, x2, x3, x4,bem_out], dim=1)
        #boundary_prediction = self.classifier(x)
        boundary_prediction1 = self.classifier2(x2)
        #boundary_prediction = self.activation(boundary_prediction)
        boundary_prediction1 = self.activation(boundary_prediction1)
        return boundary_prediction1, boundary_prediction1
class BoundaryEnhancementModule(nn.Module):
    def __init__(self, in_chs=1, out_chs=256):
        super(BoundaryEnhancementModule, self).__init__()
        self.horizontal_conv = nn.Sequential(
            nn.Conv2d(in_chs, 128, (1, 7)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )  # bs,1,256,256
        self.conv1x1_h = nn.Conv2d(2, 8, 1)

        self.vertical_conv = nn.Sequential(
            nn.Conv2d(in_chs, 128, (7, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
        self.conv1x1_v = nn.Conv2d(2, 8, 1)
        self.conv_out = nn.Conv2d(16, out_chs, 1)

    def forward(self, x):
        bs, chl, w, h = x.size()
        x_h = self.horizontal_conv(x)
        x_h = F.interpolate(x_h, size=(w, h), mode='bilinear', align_corners=True)
        x_v = self.vertical_conv(x)
        x_v = F.interpolate(x_v, size=(w, h), mode='bilinear', align_corners=True)
        x_arr = x.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        canny = np.zeros((bs, 1, w, h))
        for i in range(bs):
            canny[i] = cv2.Canny(x_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()

        h_canny = torch.cat((x_h, canny), dim=1)
        v_canny = torch.cat((x_v, canny), dim=1)
        h_v_canny = torch.cat((self.conv1x1_h(h_canny), self.conv1x1_v(v_canny)), dim=1)
        h_v_canny_out = self.conv_out(h_v_canny)

        return h_v_canny_out
class Conv3x3(nn.Module):
    def __init__(self, in_chs, out_chs, dilation=1, dropout=None):
        super(Conv3x3, self).__init__()
        if dropout is None:
            self.conv = ModuleHelper.Conv3x3_BNReLU(in_channels=in_chs, out_channels=out_chs, dilation=dilation)
        else:
            self.conv = nn.Sequential(
                ModuleHelper.Conv3x3_BNReLU(in_channels=in_chs, out_channels=out_chs, dilation=dilation),
                nn.Dropout(dropout)
            )
        initialize_weights(self.conv)
    def forward(self, x):
        return self.conv(x)
class ModuleHelper:

    @staticmethod

    def BNReLU(num_features, inplace=True):

        return nn.Sequential(

            nn.BatchNorm2d(num_features),

            nn.ReLU(inplace=inplace)

        )



    @staticmethod

    def BatchNorm2d(num_features):

        return nn.BatchNorm2d(num_features)



    @staticmethod

    def Conv3x3_BNReLU(in_channels, out_channels, stride=1, dilation=1, groups=1):

        return nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,

                      groups=groups, bias=False),

            ModuleHelper.BNReLU(out_channels)

        )



    @staticmethod

    def Conv1x1_BNReLU(in_channels, out_channels):

        return nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),

            ModuleHelper.BNReLU(out_channels)

        )



    @staticmethod

    def Conv1x1(in_channels, out_channels):

        return nn.Conv2d(in_channels, out_channels, kernel_size=1)





class Conv3x3(nn.Module):

    def __init__(self, in_chs, out_chs, dilation=1, dropout=None):

        super(Conv3x3, self).__init__()


        if dropout is None:

            self.conv = ModuleHelper.Conv3x3_BNReLU(in_channels=in_chs, out_channels=out_chs, dilation=dilation)

        else:

            self.conv = nn.Sequential(

                ModuleHelper.Conv3x3_BNReLU(in_channels=in_chs, out_channels=out_chs, dilation=dilation),

                nn.Dropout(dropout)

            )

    def forward(self, x):

        return self.conv(x)
class DecoderBlock(nn.Module):
    def __init__(self, in_chs, out_chs, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.doubel_conv = nn.Sequential(
            Conv3x3(in_chs=in_chs, out_chs=out_chs, dropout=dropout),
            Conv3x3(in_chs=out_chs, out_chs=out_chs, dropout=dropout)
        )
        

    def forward(self, x):
        out = self.doubel_conv(x)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class DoubleAttentionModule(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(DoubleAttentionModule, self).__init__()
        self.db = DecoderBlock(in_chs, out_chs)
        self.ca = ChannelAttention(out_chs)
        self.sa = SpatialAttention()

    def forward(self, x):
        out_db = self.db(x)
        out_ca = self.ca(out_db) * out_db
        out_sa = self.sa(out_db) * out_db
        return out_ca + out_sa

class BoundaryUNet(nn.Module):
    def __init__(self,  in_chns, class_num):
        super(BoundaryUNet, self).__init__()
        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.edge_module = Edge_Module()
        self.boundary_predictor = nn.Sequential(
            nn.Conv2d(params['feature_chns'][0], 1, kernel_size=1), 
            nn.Sigmoid()
        )
        self.params = params
        self.DA0 = DoubleAttentionModule(64, 128)
        self.DA1 = DoubleAttentionModule(32, 128)

    def forward(self, x):
        encoded_features = self.encoder(x) 
        segmentation_output,fts,rep,decoded_features = self.decoder(encoded_features)
        da0 = self.DA0(decoded_features[1])
        da1 = self.DA1(decoded_features[2])
        boundary_prediction,boundary_prediction1  = self.edge_module(decoded_features,x,da0,da1)
        boundary_prediction2 = self.boundary_predictor(encoded_features[0])
        return segmentation_output,fts,rep,boundary_prediction,boundary_prediction1,boundary_prediction2