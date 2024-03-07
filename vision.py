import torch
import torch.nn as nn
from tensorboardX import SummaryWriter as Summarywriter

import torch.nn.functional as F
from lib.decoders import CASCADE_Add
from lib.maxxvit_4out import maxxvit_rmlp_small_rw_256 as maxxvit_rmlp_small_rw_256_4out


def load_pretrained_weights(img_size, model_scale):
    backbone = maxxvit_rmlp_small_rw_256_4out()
    state_dict = torch.load('D:\Medical_iamge\project_store\MaxViT_ScConv_Mutiattn_MUTION\pretrained_pth\maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
    backbone.load_state_dict(state_dict, strict=False)

class MERIT_Parallel(nn.Module):  # todo 主干网络
    def __init__(self, n_class=1, img_size_s1=(256, 256), img_size_s2=(224, 224), model_scale='small',
                 decoder_aggregation='additive', interpolation='bilinear'):
        super(MERIT_Parallel, self).__init__()

        self.n_class = n_class
        self.img_size_s1 = img_size_s1
        self.img_size_s2 = img_size_s2
        self.model_scale = model_scale
        self.decoder_aggregation = decoder_aggregation
        self.interpolation = interpolation

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # backbone network initialization with pretrained weight
        self.backbone1 = load_pretrained_weights(self.img_size_s1[0], self.model_scale)
        # self.backbone2 = load_pretrained_weights(self.img_size_s2[0], self.model_scale)

        if (self.model_scale == 'tiny'):
            self.channels = [512, 256, 128, 64]
        elif (self.model_scale == 'small'):
            self.channels = [768, 384, 192, 96]

        # shared decoder initialization
        # if (self.decoder_aggregation == 'additive'):
        self.decoder = CASCADE_Add(channels=self.channels)
        # elif (self.decoder_aggregation == 'concatenation'):
        #     self.decoder = CASCADE_Cat(channels=self.channels)
        # else:
        #     sys.exit(
        #         "'" + self.decoder_aggregation + "' is not a valid decoder aggregation! Currently supported aggregations are 'additive' and 'concatenation'.")

        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)

    def forward(self, x):

        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)

        # transformer backbone as encoder
        f1 = self.backbone1(F.interpolate(x, size=self.img_size_s1, mode=self.interpolation))
        # print([f1[3].shape,f1[2].shape,f1[1].shape,f1[0].shape])

        # f2 = self.backbone2(F.interpolate(x, size=self.img_size_s2, mode=self.interpolation))
        # print([f2[3].shape,f2[2].shape,f2[1].shape,f2[0].shape])

        # decoder
        x11_o, x12_o, x13_o, x14_o = self.decoder(f1[3], [f1[2], f1[1], f1[0]])

        # x21_o, x22_o, x23_o, x24_o = self.decoder(f2[3], [f2[2], f2[1], f2[0]])

        # prediction heads
        p11 = self.out_head1(x11_o)
        p12 = self.out_head2(x12_o)
        p13 = self.out_head3(x13_o)
        p14 = self.out_head4(x14_o)
        # print([p11.shape,p12.shape,p13.shape,p14.shape])

        # p21 = self.out_head1(x21_o)
        # p22 = self.out_head2(x22_o)
        # p23 = self.out_head3(x23_o)
        # p24 = self.out_head4(x24_o)
        # print([p21.shape,p22.shape,p23.shape,p24.shape])

        p11 = F.interpolate(p11, scale_factor=32, mode=self.interpolation)
        p12 = F.interpolate(p12, scale_factor=16, mode=self.interpolation)
        p13 = F.interpolate(p13, scale_factor=8, mode=self.interpolation)
        p14 = F.interpolate(p14, scale_factor=4, mode=self.interpolation)

        # p21 = F.interpolate(p21, size=(p11.shape[-2:]), mode=self.interpolation)
        # p22 = F.interpolate(p22, size=(p12.shape[-2:]), mode=self.interpolation)
        # p23 = F.interpolate(p23, size=(p13.shape[-2:]), mode=self.interpolation)
        # p24 = F.interpolate(p24, size=(p14.shape[-2:]), mode=self.interpolation)

        # p1 = p11 + p21
        # p2 = p12 + p22
        # p3 = p13 + p23
        # p4 = p14 + p24
        # print([p1.shape,p2.shape,p3.shape,p4.shape])

        # ===================zmx
        p1 = p11
        p2 = p12
        p3 = p13
        p4 = p14
        # ===================zmx

        return p1, p2, p3, p4
        # return p1  # zmx todo

net = MERIT_Parallel()
img = torch.rand(1,3,256,256)
with Summarywriter(logdir='logs') as w:
    w.add_graph(net,img)
