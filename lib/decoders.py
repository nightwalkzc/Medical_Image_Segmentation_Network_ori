import torch
import torch.nn as nn
import torch.nn.functional as F



class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        #print(x.shape)
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=3, stride=1, padding=1, groups=1):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=kernel_size,stride=stride,padding=padding,bias=True),
		    nn.BatchNorm2d(ch_out),
	            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

        
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
    
class Attention_block_2(nn.Module):
    def __init__(self, F_g, F_l,F_int):
        super(Attention_block_2, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_g_4 = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g_4,g_3, x):
        g_4 = self.W_g(g_4)
        x1 = self.W_x(x)
        g_3 = self.W_g(g_3)
        psi = self.relu(g_4 + g_3 + x1)
        psi = self.psi(psi)

        return x * psi
    

class Attention_block_1(nn.Module):
    def __init__(self, F_g, F_l,F_int):
        super(Attention_block_1, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_g_4 = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g_4,g_3,g_2, x):
        g_4 = self.W_g(g_4)
        x1 = self.W_x(x)
        g_3 = self.W_g(g_3)
        g_2 = self.W_g(g_2)
        psi = self.relu(g_4 + g_3+ g_2 + x1)
        psi = self.psi(psi)

        return x * psi



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        #print(x.shape)
        max_pool_out= self.max_pool(x) #torch.topk(x,3, dim=1).values

        max_out = self.fc2(self.relu1(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out) 

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)    
    
    # add ScConv
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        # x = x.view(N, self.group_num, -1)
        x = x.reshape(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


class CASCADE_Cat(nn.Module):
    def __init__(self, channels=[512,320,128,64]):
        super(CASCADE_Cat,self).__init__()
        
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
	
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=2*channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=2*channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]/2))
        self.ConvBlock1 = conv_block(ch_in=2*channels[3], ch_out=channels[3])
        
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(2*channels[1])
        self.CA2 = ChannelAttention(2*channels[2])
        self.CA1 = ChannelAttention(2*channels[3])
        
        self.SA = SpatialAttention()
      
    def forward(self,x, skips):
    
        d4 = self.Conv_1x1(x)
        
        # CAM4
        d4 = self.CA4(d4)*d4
        d4 = self.SA(d4)*d4 
        d4 = self.ConvBlock4(d4)
        
        # upconv3
        d3 = self.Up3(d4)
        
        # AG3
        x3 = self.AG3(g=d3,x=skips[0])
        
        # Concat 3
        d3 = torch.cat((x3,d3),dim=1)
        
        # CAM3
        d3 = self.CA3(d3)*d3
        d3 = self.SA(d3)*d3        
        d3 = self.ConvBlock3(d3)
        
        # upconv2
        d2 = self.Up2(d3)
        
        # AG2
        x2 = self.AG2(g=d2,x=skips[1])
        
        # Concat 2
        d2 = torch.cat((x2,d2),dim=1)
        
        # CAM2
        d2 = self.CA2(d2)*d2
        d2 = self.SA(d2)*d2
        #print(d2.shape)
        d2 = self.ConvBlock2(d2)
        
        # upconv1
        d1 = self.Up1(d2)
        
        #print(skips[2])
        # AG1
        x1 = self.AG1(g=d1,x=skips[2])
        
        # Concat 1
        d1 = torch.cat((x1,d1),dim=1)
        
        # CAM1
        d1 = self.CA1(d1)*d1
        d1 = self.SA(d1)*d1
        d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1
        

class CASCADE_Add(nn.Module):  
    def __init__(self, channels=[512,320,128,64]):
        super(CASCADE_Add,self).__init__()
        
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
	
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]/2))
        self.ConvBlock1 = conv_block(ch_in=channels[3], ch_out=channels[3])
        
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(channels[1])
        self.CA2 = ChannelAttention(channels[2])
        self.CA1 = ChannelAttention(channels[3])
        
        self.SA = SpatialAttention()

        # todo 
        self.Scc1 = ScConv(768)
        self.Scc2 = ScConv(384)
        self.Scc3 = ScConv(192)
        self.Scc4 = ScConv(96)
        self.Conv_1x1_1 = nn.Conv2d(channels[1], channels[1], kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_2 = nn.Conv2d(channels[2], channels[2], kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_3 = nn.Conv2d(channels[3], channels[3], kernel_size=1, stride=1, padding=0)
      
    def forward(self,x, skips):
        a = 0.1
        b = 0.2
        c = 0.3 
        d = 0.4
        e = 0.5
    
        d4 = self.Conv_1x1(x)

        # upconv 相关添加
        d3_4 = self.Up3(x)
        d2_4 = self.Up2(d3_4)
        d1_4 = self.Up1(d2_4)
        d2_3 = self.Up2(skips[0])
        d1_3 = self.Up1(d2_3)
        d1_2 = self.Up1(skips[1])
        # upconv 相关添加

        # ScConv 添加相关
        d4_1_sc = self.Scc1(d4)
        d4 = d4_1_sc+d4
        # ScConv 添加相关


        # CAM4
        # attention:在不添加ScConv的代码中不使用该1×1卷积
        # d4 = self.ConvBlock4(d4)
        
        # upconv3
        d3 = self.Up3(d4)


        # AG3
        x3 = self.AG3(g=d3,x=skips[0]+d3_4)

        # ScConv 添加相关
        y3 = self.Conv_1x1_1(x3)
        d3_1_sc = self.Scc2(y3)
        d3 = d3_1_sc+x3  
        # ScConv 添加相关
        
        # aggregate 3
        d3 = d3 + x3  # 这里的不同concat方式和不同的1×1卷积可以做几组对照实现 | 例：把SC放在残差之前or之后
        
        # CAM3    
        # d3 = self.ConvBlock3(d3)
        
        # upconv2
        d2 = self.Up2(d3)
        
        # AG2
        # x2 = self.AG2(g=d2,x=skips[1])
        x2 = self.AG2(g=d2,x=skips[1]+d2_3+d2_4)

        # ScConv 添加相关
        y2 = self.Conv_1x1_2(x2)
        d2_1_sc = self.Scc3(y2)
        d2 = d2_1_sc+x2  
        # ScConv 添加相关
        
        # aggregate 2
        d2 = d2 + x2
        
        # CAM2
        # d2 = self.ConvBlock2(d2)
        
        # upconv1
        d1 = self.Up1(d2)
        
        #print(skips[2])
        # AG1
        x1 = self.AG1(g=d1,x=skips[2]+d1_2+d1_3+d1_4)


        # ScConv 添加相关
        y1 = self.Conv_1x1_3(x1)
        d1_1_sc = self.Scc4(y1)
        d1 = d1_1_sc+x1  
        # ScConv 添加相关
        
        # aggregate 1
        d1 = d1 + x1
        
        # CAM1
        # d1 = self.CA1(d1)*d1
        # d1 = self.SA(d1)*d1
        # d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1

class decoder_nocascaded(nn.Module):  
    def __init__(self, channels=[512,320,128,64]):
        super(decoder_nocascaded,self).__init__()
        
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
	
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]/2))
        self.ConvBlock1 = conv_block(ch_in=channels[3], ch_out=channels[3])
        
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(channels[1])
        self.CA2 = ChannelAttention(channels[2])
        self.CA1 = ChannelAttention(channels[3])
        
        self.SA = SpatialAttention()

        # todo 
        self.Scc1 = ScConv(768)
        self.Scc2 = ScConv(384)
        self.Scc3 = ScConv(192)
        self.Scc4 = ScConv(96)
        self.Conv_1x1_1 = nn.Conv2d(channels[1], channels[1], kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_2 = nn.Conv2d(channels[2], channels[2], kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_3 = nn.Conv2d(channels[3], channels[3], kernel_size=1, stride=1, padding=0)
      
    def forward(self,x, skips):

        d4 = self.Conv_1x1(x)


        # ScConv 添加相关
        d4_1_sc = self.Scc1(d4)
        d4 = d4_1_sc+d4


        # CAM4
        # attention:在不添加ScConv的代码中不使用该1×1卷积
        # d4 = self.ConvBlock4(d4)
        
        # upconv3
        d3 = self.Up3(d4)


        # AG3
        x3 = self.AG3(g=d3,x=skips[0])

        # ScConv 添加相关
        y3 = self.Conv_1x1_1(x3)
        d3_1_sc = self.Scc2(y3)
        d3 = d3_1_sc+x3  
        # ScConv 添加相关
        
        # aggregate 3
        d3 = d3 + x3  # 这里的不同concat方式和不同的1×1卷积可以做几组对照实现 | 例：把SC放在残差之前or之后
        
        # CAM3    
        # d3 = self.ConvBlock3(d3)
        
        # upconv2
        d2 = self.Up2(d3)
        
        # AG2
        # x2 = self.AG2(g=d2,x=skips[1])
        x2 = self.AG2(g=d2,x=skips[1])

        # ScConv 添加相关
        y2 = self.Conv_1x1_2(x2)
        d2_1_sc = self.Scc3(y2)
        d2 = d2_1_sc+x2  
        # ScConv 添加相关
        
        # aggregate 2
        d2 = d2 + x2
        
        # CAM2
        # d2 = self.ConvBlock2(d2)
        
        # upconv1
        d1 = self.Up1(d2)
        
        #print(skips[2])
        # AG1
        x1 = self.AG1(g=d1,x=skips[2])


        # ScConv 添加相关
        y1 = self.Conv_1x1_3(x1)
        d1_1_sc = self.Scc4(y1)
        d1 = d1_1_sc+x1  
        # ScConv 添加相关
        
        # aggregate 1
        d1 = d1 + x1
        
        # CAM1
        # d1 = self.CA1(d1)*d1
        # d1 = self.SA(d1)*d1
        # d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1
    


class decoder_nores(nn.Module):  
    def __init__(self, channels=[512,320,128,64]):
        super(decoder_nores,self).__init__()
        
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
	
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]/2))
        self.ConvBlock1 = conv_block(ch_in=channels[3], ch_out=channels[3])
        
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(channels[1])
        self.CA2 = ChannelAttention(channels[2])
        self.CA1 = ChannelAttention(channels[3])
        
        self.SA = SpatialAttention()

        # todo 
        self.Scc1 = ScConv(768)
        self.Scc2 = ScConv(384)
        self.Scc3 = ScConv(192)
        self.Scc4 = ScConv(96)
        self.Conv_1x1_1 = nn.Conv2d(channels[1], channels[1], kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_2 = nn.Conv2d(channels[2], channels[2], kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_3 = nn.Conv2d(channels[3], channels[3], kernel_size=1, stride=1, padding=0)
      
    def forward(self,x, skips):

    
        d4 = self.Conv_1x1(x)

        # upconv 相关添加
        d3_4 = self.Up3(x)
        d2_4 = self.Up2(d3_4)
        d1_4 = self.Up1(d2_4)
        d2_3 = self.Up2(skips[0])
        d1_3 = self.Up1(d2_3)
        d1_2 = self.Up1(skips[1])
        # upconv 相关添加

        # ScConv 添加相关
        d4_1_sc = self.Scc1(d4)
        d4 = d4_1_sc+d4
        # ScConv 添加相关


        # CAM4
        # attention:在不添加ScConv的代码中不使用该1×1卷积
        # d4 = self.ConvBlock4(d4)
        
        # upconv3
        d3 = self.Up3(d4)


        # AG3
        x3 = self.AG3(g=d3,x=skips[0]+d3_4)

        # ScConv 添加相关
        y3 = self.Conv_1x1_1(x3)
        d3_1_sc = self.Scc2(y3)
        d3 = d3_1_sc+x3  
        # ScConv 添加相关
        
        
        # upconv2
        d2 = self.Up2(d3)
        
        # AG2
        # x2 = self.AG2(g=d2,x=skips[1])
        x2 = self.AG2(g=d2,x=skips[1]+d2_3+d2_4)

        # ScConv 添加相关
        y2 = self.Conv_1x1_2(x2)
        d2_1_sc = self.Scc3(y2)
        d2 = d2_1_sc+x2  
        # ScConv 添加相关

        
        # CAM2
        # d2 = self.ConvBlock2(d2)
        
        # upconv1
        d1 = self.Up1(d2)
        
        #print(skips[2])
        # AG1
        x1 = self.AG1(g=d1,x=skips[2]+d1_2+d1_3+d1_4)


        # ScConv 添加相关
        y1 = self.Conv_1x1_3(x1)
        d1_1_sc = self.Scc4(y1)
        d1 = d1_1_sc+x1  
        # ScConv 添加相关
        
        
        # CAM1
        # d1 = self.CA1(d1)*d1
        # d1 = self.SA(d1)*d1
        # d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1
    

class decoder_nosc_up(nn.Module):  
    def __init__(self, channels=[512,320,128,64]):
        super(decoder_nosc_up,self).__init__()
        
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
	
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]/2))
        self.ConvBlock1 = conv_block(ch_in=channels[3], ch_out=channels[3])
        
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(channels[1])
        self.CA2 = ChannelAttention(channels[2])
        self.CA1 = ChannelAttention(channels[3])
        
        self.SA = SpatialAttention()

        # todo 
        self.Scc1 = ScConv(768)
        self.Scc2 = ScConv(384)
        self.Scc3 = ScConv(192)
        self.Scc4 = ScConv(96)
        self.Conv_1x1_1 = nn.Conv2d(channels[1], channels[1], kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_2 = nn.Conv2d(channels[2], channels[2], kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_3 = nn.Conv2d(channels[3], channels[3], kernel_size=1, stride=1, padding=0)
      
    def forward(self,x, skips):

    
        d4 = self.Conv_1x1(x)

        # upconv 相关添加
        d3_4 = self.Up3(x)
        d2_4 = self.Up2(d3_4)
        d1_4 = self.Up1(d2_4)
        d2_3 = self.Up2(skips[0])
        d1_3 = self.Up1(d2_3)
        d1_2 = self.Up1(skips[1])
        # upconv 相关添加

        # ScConv 添加相关
        d4_1_sc = self.Scc1(d4)
        d4 = d4_1_sc+d4
        # ScConv 添加相关


        # CAM4
        # attention:在不添加ScConv的代码中不使用该1×1卷积
        # d4 = self.ConvBlock4(d4)
        
        # upconv3
        # d3 = self.Up3(d4)


        # AG3
        x3 = self.AG3(g=d3_4,x=skips[0])

        # ScConv 添加相关
        y3 = self.Conv_1x1_1(x3)
        d3_1_sc = self.Scc2(y3)
        d3 = d3_1_sc+x3  
        # ScConv 添加相关
        
        
        # upconv2
        d2 = self.Up2(d3)
        
        # AG2
        # x2 = self.AG2(g=d2,x=skips[1])
        x2 = self.AG2(g=d2_3+d2_4,x=skips[1])

        # ScConv 添加相关
        y2 = self.Conv_1x1_2(x2)
        d2_1_sc = self.Scc3(y2)
        d2 = d2_1_sc+x2  
        # ScConv 添加相关

        
        # CAM2
        # d2 = self.ConvBlock2(d2)
        
        # upconv1
        d1 = self.Up1(d2)
        
        #print(skips[2])
        # AG1
        x1 = self.AG1(g=d1_2+d1_3+d1_4,x=skips[2])


        # ScConv 添加相关
        y1 = self.Conv_1x1_3(x1)
        d1_1_sc = self.Scc4(y1)
        d1 = d1_1_sc+x1  
        # ScConv 添加相关
        
        
        # CAM1
        # d1 = self.CA1(d1)*d1
        # d1 = self.SA(d1)*d1
        # d1 = self.ConvBlock1(d1)
        return d4, d3, d2, d1
    
class decoder_mutiattn_mution(nn.Module):  
    def __init__(self, channels=[512,320,128,64]):
        super(decoder_mutiattn_mution,self).__init__()
        
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
	
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        # self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.AG2 = Attention_block_2(F_g=channels[2], F_l=channels[2] ,F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=channels[2], ch_out=channels[2])
        
        self.Up1 = up_conv(ch_in=channels[2],ch_out=channels[3])
        # self.AG1 = Attention_block(F_g=channels[3],F_l=channels[3],F_int=int(channels[3]/2))
        self.AG1 = Attention_block_1(F_g=channels[3], F_l=channels[3], F_int=int(channels[3]/2))
        self.ConvBlock1 = conv_block(ch_in=channels[3], ch_out=channels[3])
        
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(channels[1])
        self.CA2 = ChannelAttention(channels[2])
        self.CA1 = ChannelAttention(channels[3])
        
        self.SA = SpatialAttention()

        # todo 
        self.Scc1 = ScConv(768)
        self.Scc2 = ScConv(384)
        self.Scc3 = ScConv(192)
        self.Scc4 = ScConv(96)
        self.Conv_1x1_1 = nn.Conv2d(channels[1], channels[1], kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_2 = nn.Conv2d(channels[2], channels[2], kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_3 = nn.Conv2d(channels[3], channels[3], kernel_size=1, stride=1, padding=0)
      
    def forward(self,x, skips):

    


        # upconv 相关添加
        d3_4 = self.Up3(x)
        d2_4 = self.Up2(d3_4)
        d1_4 = self.Up1(d2_4)
        d2_3 = self.Up2(skips[0])
        d1_3 = self.Up1(d2_3)
        d1_2 = self.Up1(skips[1])
        # upconv 相关添加

        # ScConv 添加相关
        # d4 = self.Conv_1x1(x)
        # d4_1_sc = self.Scc1(d4)
        # d4 = d4_1_sc+d4

        d4 = x

        # AG3
        x3 = self.AG3(g=d3_4,x=skips[0])

        # ScConv 添加相关
        y3 = self.Conv_1x1_1(x3)
        d3_1_sc = self.Scc2(y3)
        d3 = d3_1_sc+x3  
        # ScConv 添加相关
        
        
        
        # AG2
        # x2 = self.AG2(g=d2,x=skips[1])
        # x2 = self.AG2(g=d2_3+d2_4,x=skips[1])
        x2 = self.AG2(g_4=d2_4,g_3=d2_3,x=skips[1])

        # ScConv 添加相关
        y2 = self.Conv_1x1_2(x2)
        d2_1_sc = self.Scc3(y2)
        d2 = d2_1_sc+x2  
        # ScConv 添加相关

        
        # CAM2
        # d2 = self.ConvBlock2(d2)

        
        #print(skips[2])
        # AG1
        # x1 = self.AG1(g=d1_2+d1_3+d1_4,x=skips[2])
        x1 = self.AG1(g_4=d1_4,g_3=d1_3,g_2=d1_2,x=skips[2])


        # ScConv 添加相关
        y1 = self.Conv_1x1_3(x1)
        d1_1_sc = self.Scc4(y1)
        d1 = d1_1_sc+x1  
        # ScConv 添加相关

        # d1 = self.ConvBlock1(d1)

        return d4, d3, d2, d1
