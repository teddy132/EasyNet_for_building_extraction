import torch
import torch.nn as nn
import torch.nn.functional as F

#from ..base import modules as md
from functools import partial
import math
nonlinearity = partial(F.relu,inplace=True)
import numpy as np
import torch.nn as nn
import timm.models
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import ClassifierHead, AvgPool2dSame, ConvBnAct, SEModule, DropPath,EffectiveSEModule
from timm.models.registry import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)



class CTF_block(nn.Module):
    """CNN and Transformer Fusion Block
    """

    def __init__(self, in_channel, out_channel,act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2d,norm_layer2=nn.LayerNorm,img_size=128,
                patch_size=3,
                in_chans=3,
                num_classes=1000,
                #embed_dims=[64, 128, 256, 512],
                num_heads=4,
                mlp_ratios=4,
                qkv_bias=False,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,

                depths=1,
                sr_ratios=2,):
        super(CTF_block, self).__init__()


        cargs = dict(act_layer=act_layer, norm_layer=norm_layer)

        self.CNN_Contact = Commondecoder(in_channel,out_channel)
        self.patch_embed = OverlapPatchEmbed(
            img_size=img_size, patch_size=3, stride=2, in_chans=in_channel, embed_dim=out_channel
        )
        self.mixblock = nn.ModuleList(
            [
                Block(
                    dim=out_channel,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    # drop_path=dpr[cur + i],
                    norm_layer=norm_layer2,
                    sr_ratio=sr_ratios,
                )
                for i in range(depths)
            ]
        )
        self.norm1 = norm_layer2(out_channel)




    def forward(self, x):

        x1 = self.CNN_Contact(x)

        B = x.shape[0]#batch size
        x2, H, W = self.patch_embed(x)
        for i, blk in enumerate(self.mixblock):
            x2 = blk(x2, H, W)
        x2 = self.norm1(x2)
        x2 = x2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x2 = F.interpolate(x2, scale_factor=2, mode="nearest")
        x = x1+x2

        return x


class Easy_Net(nn.Module):

    # pylint: disable=unused-variable
    def __init__(self, block, layers, encoder_channels,
                 decoder_channels, dilation=1,dropblock_prob=0,
                 last_gamma=True, norm_layer=nn.BatchNorm2d,center=False,se_ratio=0.25):


        self.last_gamma = last_gamma
        super(Easy_Net, self).__init__()
        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        out_channels =decoder_channels

        self.layer1 = self._make_layer(block,encoder_channels[0], encoder_channels[1], layers[0], norm_layer=norm_layer, stride=1)
        self.layer2 = self._make_layer(block,encoder_channels[1], out_channels[0], layers[1], stride=1, norm_layer=norm_layer,dilation=1)
        #self.layer2 = CTF_block1(encoder_channels[1], out_channels[0])

        self.layer3 = self._make_layer(block,encoder_channels[2], encoder_channels[1], layers[2], stride=1,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)

        self.layer4 = self._make_layer(block, encoder_channels[-1],out_channels[0], layers[3], stride=1,
                                       norm_layer=norm_layer,
                                       dropblock_prob=dropblock_prob)
        self.last_cont1 = CTF_block(out_channels[0] * 2, out_channels[-2])
        self.last_cont1 = Commondecoder(out_channels[0] * 2, out_channels[-2])
        self.last_cont2 = Commondecoder(out_channels[-2], out_channels[-1])



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block,inplanes, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True):

        layers = []
        layers.append(block(inplanes, planes,
                            ))
        for i in range(1, blocks):
            layers.append(block(planes, planes,
                               ))

        return nn.Sequential(*layers)

    def forward(self, *features):
        features = features[1:]  # remove first skip with same spatial resolution
        #features = features[::-1]  # reverse channels to start from head of encoder

        s1,s2,s3,s4 = features[0],features[1],features[2],features[3]
        s1 = self.layer1(s1)
        s1 = F.interpolate(s1, scale_factor=0.5, mode="nearest")

        s3 = self.layer3(s3)
        s3 = F.interpolate(s3, scale_factor=2, mode="nearest")

        x1 = s1+s2+s3
        x1 = self.layer2(x1)

        x2 = self.layer4(s4)
        x2 = F.interpolate(x2, scale_factor=4, mode="nearest")
        x = torch.cat([x1, x2], dim=1)

        x = self.last_cont1(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.last_cont2(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        return x




class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        #  self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        #  atten = self.sigmoid_atten(atten)
        atten = atten.sigmoid()
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class Commondecoder(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self, inplanes, planes, act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2d):
        super(Commondecoder, self).__init__()


        cargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        self.conv1 = ConvBnAct(inplanes, planes, kernel_size=3, **cargs)
        self.conv2 = ConvBnAct(
            planes, planes, kernel_size=3,  **cargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class encoder_predict(nn.Module):

    # pylint: disable=unused-variable
    def __init__(self, encoder_channels,
                 decoder_channels, norm_layer=nn.BatchNorm2d):


        self.last_gamma = True
        super(encoder_predict, self).__init__()
        self.trans = ConvBnAct(in_channels=encoder_channels[-1],out_channels=16)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, *features):
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        x = features[0]
        x = self.trans(x)
        x = F.interpolate(x, scale_factor=4, mode="nearest")
        return x



#学习位置的模块，Mix_FFN
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


