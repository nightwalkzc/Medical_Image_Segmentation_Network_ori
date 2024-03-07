import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import timm

import logging

from scipy import ndimage

from lib.decoders import CASCADE_Add, CASCADE_Cat,decoder_nocascaded,decoder_nores,decoder_nosc_up,decoder_mutiattn_mution
# from lib.decoders_2 import CASCADE_Add
# from lib.decoders_1 import CASCADE_Add

from lib.maxxvit_4out import maxvit_tiny_rw_224 as maxvit_tiny_rw_224_4out
from lib.maxxvit_4out import maxvit_rmlp_tiny_rw_256 as maxvit_rmlp_tiny_rw_256_4out
from lib.maxxvit_4out import maxxvit_rmlp_small_rw_256 as maxxvit_rmlp_small_rw_256_4out
from lib.maxxvit_4out import maxvit_rmlp_small_rw_224 as maxvit_rmlp_small_rw_224_4out


logger = logging.getLogger(__name__)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def load_pretrained_weights(img_size, model_scale):
    
    if(model_scale=='tiny'):
        if img_size==224:
            backbone = maxvit_tiny_rw_224_4out()  # [64, 128, 320, 512]
            print('Loading:', './pretrained_pth/maxvit/maxvit_tiny_rw_224_sw-7d0dffeb.pth')
            state_dict = torch.load('./pretrained_pth/maxvit/maxvit_tiny_rw_224_sw-7d0dffeb.pth')
        elif(img_size==256):
            backbone = maxvit_rmlp_tiny_rw_256_4out()
            print('Loading:', './pretrained_pth/maxvit/maxvit_rmlp_tiny_rw_256_sw-bbef0ff5.pth')
            state_dict = torch.load('./pretrained_pth/maxvit/maxvit_rmlp_tiny_rw_256_sw-bbef0ff5.pth')
        else:
            sys.exit(str(img_size)+" is not a valid image size! Currently supported image sizes are 224 and 256.")
    elif(model_scale=='small'):
        if img_size==224:
            backbone = maxvit_rmlp_small_rw_224_4out()  # [64, 128, 320, 512]  这里这里
            print('Loading:', './pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
            state_dict = torch.load('./pretrained_pth/maxvit/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
        elif(img_size==256):
            backbone = maxxvit_rmlp_small_rw_256_4out()
            print('Loading:', './pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
            # state_dict = torch.load('./pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
            state_dict = torch.load('F:\project_store\MaxViT_ScConv_Mutiattn_MUTION\pretrained_pth\maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
        else:
            sys.exit(str(img_size)+" is not a valid image size! Currently supported image sizes are 224 and 256.")
    else:
        sys.exit(model_scale+" is not a valid model scale! Currently supported model scales are 'tiny' and 'small'.")
        
    backbone.load_state_dict(state_dict, strict=False)
    print('Pretrain weights loaded.')
    
    return backbone


class MERIT_Parallel(nn.Module): 
    def __init__(self, n_class=1, img_size_s1=(256,256), img_size_s2=(224,224), model_scale='small', decoder_aggregation='additive', interpolation='bilinear'):
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
        
        if(self.model_scale=='tiny'):
            self.channels = [512, 256, 128, 64]
        elif(self.model_scale=='small'):
            self.channels = [768, 384, 192, 96]
     
        # shared decoder initialization
        if(self.decoder_aggregation=='additive'):
            self.decoder = decoder_mutiattn_mution(channels=self.channels)
        elif(self.decoder_aggregation=='concatenation'):
            i = 521
            # self.decoder = CASCADE_Cat(channels=self.channels)
        else:
            sys.exit("'"+self.decoder_aggregation+"' is not a valid decoder aggregation! Currently supported aggregations are 'additive' and 'concatenation'.")
        
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
        #print([f1[3].shape,f1[2].shape,f1[1].shape,f1[0].shape])
          
        # decoder
        x11_o, x12_o, x13_o, x14_o = self.decoder(f1[3], [f1[2], f1[1], f1[0]])
        
        
        # prediction heads  
        p11 = self.out_head1(x11_o)
        p12 = self.out_head2(x12_o)
        p13 = self.out_head3(x13_o)
        p14 = self.out_head4(x14_o)
        #print([p11.shape,p12.shape,p13.shape,p14.shape])
              
        p11 = F.interpolate(p11, scale_factor=32, mode=self.interpolation)
        p12 = F.interpolate(p12, scale_factor=16, mode=self.interpolation)
        p13 = F.interpolate(p13, scale_factor=8, mode=self.interpolation)
        p14 = F.interpolate(p14, scale_factor=4, mode=self.interpolation)
        
        

        # ===================zmx
        p1 = p11
        p2 = p12
        p3 = p13
        p4 = p14
        # ===================zmx

        return p1, p2, p3, p4

