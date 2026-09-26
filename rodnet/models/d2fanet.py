import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import to_2tuple
import torch.fft as fft
from torch.fft import ifftshift
from rodnet.ops.dcn import SpatialDeformConvPack3D, TemporalDeformConvPack3D, DeformConvPack3D, DeformConvPack3D_True



class ContextBranch(nn.Module):
    def __init__(self, in_channels, depths=[2, 2, 4, 2], channels=[64, 128, 256, 512], drop_rate=0.1, tokenMixer=['c', 'c', 'c', 'c']):
        super(ContextBranch, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=channels[0], kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2), bias=False),
            nn.GroupNorm(channels[0] // 4, channels[0]),
        )

        tokenMixer_list = []
        for i in tokenMixer:
            if i == 'c':
                tokenMixer_list.append(ConvFormerBlock)
            elif i == 'a':
                tokenMixer_list.append(AttentionFormerBlock)
            else:
                raise TypeError

        self.encoder_block1 = nn.Sequential(
            *[tokenMixer_list[0](channels[0], drop_rate)
              for _ in range(depths[0])])

        self.dowm_sample1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.GroupNorm(channels[0] // 4, channels[0]),
            nn.Conv3d(in_channels=channels[0], out_channels=channels[1], kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(inplace=True),
        )

        self.encoder_block2 = nn.Sequential(
            *[tokenMixer_list[1](channels[1], drop_rate)
              for _ in range(depths[1])])

        self.down_sample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GroupNorm(channels[1] // 4, channels[1]),
            nn.Conv3d(in_channels=channels[1], out_channels=channels[2], kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(inplace=True),
        )

        self.encoder_block3 = nn.Sequential(
            *[tokenMixer_list[2](channels[2], drop_rate)
              for _ in range(depths[2])])

        self.down_sample3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GroupNorm(channels[2] // 4, channels[2]),
            nn.Conv3d(in_channels=channels[2], out_channels=channels[3], kernel_size=1, stride=1, padding=0, bias=True),
            nn.SiLU(inplace=True),
        )

        self.encoder_block4 = nn.Sequential(
            *[tokenMixer_list[3](channels[3], drop_rate)
              for _ in range(depths[3])])

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder_block1(x1)
        x2 = self.dowm_sample1(x1)
        x2 = self.encoder_block2(x2)
        x3 = self.down_sample2(x2)
        x3 = self.encoder_block3(x3)
        x4 = self.down_sample3(x3)
        x4 = self.encoder_block4(x4)
        return x1, x2, x3, x4


class SepConv(nn.Module):
    def __init__(self, dim, expansion_ratio=4, kernel_size=(5, 7, 7), padding=(2, 3, 3)):
        super(SepConv, self).__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv3d(dim, med_channels, 1, 1, 0, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.dwconv = nn.Conv3d(med_channels, med_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=med_channels, bias=True)
        self.pwconv2 = nn.Conv3d(med_channels, dim, 1, 1, 0, bias=True)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.pwconv2(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, h, d_model):
        super(CrossAttention, self).__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.h = h
        self.q = nn.Linear(d_model, d_model, bias=True)
        self.k = nn.Linear(d_model, d_model, bias=True)
        self.v = nn.Linear(d_model, d_model, bias=True)

    @staticmethod
    def attention(q, k, v):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    def forward(self, q, k, v):
        b, _, f, r, a = q.shape
        q = q.reshape(b, self.d_model, -1).permute(0, 2, 1)
        k = k.reshape(b, self.d_model, -1).permute(0, 2, 1)
        v = v.reshape(b, self.d_model, -1).permute(0, 2, 1)
        q = self.q(q).reshape(b, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.k(k).reshape(b, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        v = self.v(v).reshape(b, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        out = self.attention(q, k, v)
        out = out.permute(0, 3, 2, 1).reshape(b, self.d_model, f, r, a)
        return out


class Mlp(nn.Module):
    def __init__(self, dim, drop_rate=0.1, mlp_ratio=4):
        super(Mlp, self).__init__()
        in_features = dim
        out_features = in_features
        hidden_features = int(mlp_ratio * in_features)
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1, 1, 0, bias=True)
        drop_probs = to_2tuple(drop_rate)
        self.drop1 = nn.Dropout(drop_probs[0])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, head_dim=32, num_heads=None):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1
        self.attention_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(self.attention_dim, dim, bias=True)

    def forward(self, x):
        B, C, F, R, A = x.shape
        N = F * R * A
        qkv = self.qkv(x.permute(0, 2, 3, 4, 1)).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, F, R, A, self.attention_dim)
        x = self.proj(x)
        return x.permute(0, 4, 1, 2, 3)


class ConvFormerBlock(nn.Module):
    def __init__(self, dim, drop_rate, token_mixer=SepConv, mlp=Mlp):
        super(ConvFormerBlock, self).__init__()
        self.norm1 = nn.GroupNorm(dim // 4, dim)
        self.token_mixer = token_mixer(dim)
        self.norm2 = nn.GroupNorm(dim // 4, dim)
        self.mlp = mlp(dim, drop_rate)

    def forward(self, x):
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AttentionFormerBlock(nn.Module):
    def __init__(self, dim, drop_rate, token_mixer=Attention, mlp=Mlp):
        super(AttentionFormerBlock, self).__init__()
        self.norm1 = nn.GroupNorm(dim // 4, dim)
        self.token_mixer = token_mixer(dim)
        self.norm2 = nn.GroupNorm(dim // 4, dim)
        self.mlp = mlp(dim, drop_rate)

    def forward(self, x):
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MNet(nn.Module):
    def __init__(self, in_chirps, out_channels, use_dcn):
        super(MNet, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels
        if use_dcn:
            self.t_conv3d = DeformConvPack3D(in_channels=2, out_channels=out_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        else:
            self.t_conv3d = nn.Conv3d(in_channels=2, out_channels=out_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=True)
        t_conv_out = math.floor((in_chirps + 2 * 1 - (3 - 1) - 1) / 2 + 1)
        self.t_maxpool = nn.MaxPool3d(kernel_size=(t_conv_out, 1, 1))

    def forward(self, x):
        batch_size, n_channels, win_size, in_chirps, h, w = x.shape
        x_out = torch.empty((batch_size, self.out_channels, win_size, h, w)).cuda()
        for win in range(win_size):
            x_win = self.t_conv3d(x[:, :, win, :, :, :])
            x_win = self.t_maxpool(x_win)
            x_win = x_win.view(batch_size, self.out_channels, w, h)
            x_out[:, :, win, ] = x_win
        return x_out


class SSFM(nn.Module):
    def __init__(self, in_chirps, out_channels):
        super(SSFM, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels
        self.v_net = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(in_chirps, 1, 1), stride=(in_chirps, 1, 1),
                      padding=0),
            nn.BatchNorm3d(int(out_channels // 2)),
            nn.GELU()
        )
        self.s_net = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(1, 3, 3), stride=1,
                      padding=(0, 1, 1)),
            nn.BatchNorm3d(int(out_channels // 2)),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(in_chirps, 1, 1))
        )
        self.merge_net = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(out_channels)),
            nn.GELU()
        )

    def forward(self, x):
        batch_size, n_channels, win_size, in_chirps, w, h = x.shape
        x_out = torch.zeros((batch_size, self.out_channels, win_size, w, h)).cuda()
        for win in range(win_size):
            x_velocity = self.v_net(x[:, :, win, :, :, :]).squeeze(2)  # (B, C/2, 128, 128)
            x_space = self.s_net(x[:, :, win, :, :, :]).squeeze(2)  # (B, C/2, 128, 128)
            x_merge = torch.cat([x_velocity, x_space], dim=1)  # (B, C, 128, 128)
            x_out[:, :, win, :, :] = self.merge_net(x_merge)
        return x_out


class simDecoupledMnet_avg(nn.Module):
    def __init__(self, in_chirps, out_channels, use_dcn, only_dcn):
        super(simDecoupledMnet_avg, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels
        self.use_dcn = use_dcn
        self.only_dcn = only_dcn

        if self.use_dcn:
            self.s_dcn = nn.Sequential(
                SpatialDeformConvPack3D(in_channels=2, out_channels=out_channels // 2, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            )
            self.t_dcn = nn.Sequential(
                TemporalDeformConvPack3D(in_channels=2, out_channels=out_channels // 2, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            )
        if self.only_dcn is False:
            self.s_conv3d = nn.Sequential(
                nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            )
            self.t_conv3d = nn.Sequential(
                nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            )
        self.s_norm_act = nn.Sequential(
            nn.GroupNorm(out_channels // 2 // 4, out_channels // 2),
            nn.SiLU(inplace=True),
        )
        self.t_norm_act = nn.Sequential(
            nn.GroupNorm(out_channels // 2 // 4, out_channels // 2),
            nn.SiLU(inplace=True),
        )

        self.m_conv3d = nn.Sequential(
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.SiLU(inplace=True),
        )
        self.avg_pool = nn.AvgPool3d(kernel_size=(in_chirps, 1, 1))

    def forward(self, x):
        b, c, win, chirps, h, w = x.shape
        x_m = torch.empty((b, self.out_channels, win, h, w)).cuda()
        for win in range(win):
            if self.only_dcn is False:
                x_win_t = self.t_conv3d(x[:, :, win, :, :, :])
                x_win_s = self.s_conv3d(x[:, :, win, :, :, :])
            if self.use_dcn:
                if self.only_dcn is False:
                    x_win_t = x_win_t + self.t_dcn(x[:, :, win, :, :, :])
                    x_win_s = x_win_s + self.s_dcn(x[:, :, win, :, :, :])
                else:
                    x_win_t = self.t_dcn(x[:, :, win, :, :, :])
                    x_win_s = self.s_dcn(x[:, :, win, :, :, :])
            x_win_t = self.t_norm_act(x_win_t)
            x_win_s = self.s_norm_act(x_win_s)
            x_win_t = self.avg_pool(x_win_t).squeeze(2)
            x_win_s = self.avg_pool(x_win_s).squeeze(2)
            x_m[:, :, win, :, :] = torch.cat([x_win_t, x_win_s], dim=1)
        out = self.m_conv3d(x_m)
        return out


class fcAvg(nn.Module):
    def __init__(self, in_chirps, out_channels):
        super(fcAvg, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.SiLU(inplace=True),
        )
        self.m_conv3d = nn.Sequential(
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.SiLU(inplace=True),
        )
        self.avg_pool = nn.AvgPool3d(kernel_size=(in_chirps, 1, 1))

    def forward(self, x):
        b, c, win, chirps, h, w = x.shape
        x_m = torch.empty((b, self.out_channels, win, h, w)).cuda()
        for win in range(win):
            x_m[:, :, win, :, :] = self.avg_pool(self.fc(x[:, :, win, :, :, :])).squeeze(2)
        out = self.m_conv3d(x_m)
        return out


class dfcAvg(nn.Module):
    def __init__(self, in_chirps, out_channels):
        super(dfcAvg, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels
        self.fc = nn.Sequential(
            DeformConvPack3D_True(2, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.SiLU(inplace=True),
        )
        self.m_conv3d = nn.Sequential(
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.SiLU(inplace=True),
        )
        self.avg_pool = nn.AvgPool3d(kernel_size=(in_chirps, 1, 1))

    def forward(self, x):
        b, c, win, chirps, h, w = x.shape
        x_m = torch.empty((b, self.out_channels, win, h, w)).cuda()
        for win in range(win):
            x_m[:, :, win, :, :] = self.avg_pool(self.fc(x[:, :, win, :, :, :])).squeeze(2)
        out = self.m_conv3d(x_m)
        return out


class simDecoupledMnet_max(nn.Module):
    def __init__(self, in_chirps, out_channels, use_dcn, only_dcn):
        super(simDecoupledMnet_max, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels
        self.use_dcn = use_dcn
        self.only_dcn = only_dcn

        if self.use_dcn:
            self.s_dcn = nn.Sequential(
                SpatialDeformConvPack3D(in_channels=2, out_channels=out_channels // 2, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            )
            self.t_dcn = nn.Sequential(
                TemporalDeformConvPack3D(in_channels=2, out_channels=out_channels // 2, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            )
        if self.only_dcn is False:
            self.s_conv3d = nn.Sequential(
                nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            )
            self.t_conv3d = nn.Sequential(
                nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            )
        self.s_norm_act = nn.Sequential(
            nn.GroupNorm(out_channels // 2 // 4, out_channels // 2),
            nn.SiLU(inplace=True),
        )
        self.t_norm_act = nn.Sequential(
            nn.GroupNorm(out_channels // 2 // 4, out_channels // 2),
            nn.SiLU(inplace=True),
        )

        self.m_conv3d = nn.Sequential(
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.SiLU(inplace=True),
        )
        self.max_pool = nn.MaxPool3d(kernel_size=(in_chirps, 1, 1))

    def forward(self, x):
        b, c, win, chirps, h, w = x.shape
        x_m = torch.empty((b, self.out_channels, win, h, w)).cuda()
        for win in range(win):
            if self.only_dcn is False:
                x_win_t = self.t_conv3d(x[:, :, win, :, :, :])
                x_win_s = self.s_conv3d(x[:, :, win, :, :, :])
            if self.use_dcn:
                if self.only_dcn is False:
                    x_win_t = x_win_t + self.t_dcn(x[:, :, win, :, :, :])
                    x_win_s = x_win_s + self.s_dcn(x[:, :, win, :, :, :])
                else:
                    x_win_t = self.t_dcn(x[:, :, win, :, :, :])
                    x_win_s = self.s_dcn(x[:, :, win, :, :, :])
            x_win_t = self.t_norm_act(x_win_t)
            x_win_s = self.s_norm_act(x_win_s)
            x_win_t = self.max_pool(x_win_t).squeeze(2)
            x_win_s = self.max_pool(x_win_s).squeeze(2)
            x_m[:, :, win, :, :] = torch.cat([x_win_t, x_win_s], dim=1)
        out = self.m_conv3d(x_m)
        return out


class simDecoupledMnet_mix(nn.Module):
    def __init__(self, in_chirps, out_channels, use_dcn, only_dcn):
        super(simDecoupledMnet_mix, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels
        self.use_dcn = use_dcn
        self.only_dcn = only_dcn

        if self.use_dcn:
            self.s_dcn = nn.Sequential(
                SpatialDeformConvPack3D(in_channels=2, out_channels=out_channels // 2, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            )
            self.t_dcn = nn.Sequential(
                TemporalDeformConvPack3D(in_channels=2, out_channels=out_channels // 2, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            )
        if self.only_dcn is False:
            self.s_conv3d = nn.Sequential(
                nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            )
            self.t_conv3d = nn.Sequential(
                nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            )
        self.s_norm_act = nn.Sequential(
            nn.GroupNorm(out_channels // 2 // 4, out_channels // 2),
            nn.SiLU(inplace=True),
        )
        self.t_norm_act = nn.Sequential(
            nn.GroupNorm(out_channels // 2 // 4, out_channels // 2),
            nn.SiLU(inplace=True),
        )

        self.m_conv3d = nn.Sequential(
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.SiLU(inplace=True),
        )
        self.max_pool = nn.MaxPool3d(kernel_size=(in_chirps, 1, 1))
        self.avg_pool = nn.AvgPool3d(kernel_size=(in_chirps, 1, 1))

    def forward(self, x):
        b, c, win, chirps, h, w = x.shape
        x_m = torch.empty((b, self.out_channels, win, h, w)).cuda()
        for win in range(win):
            if self.only_dcn is False:
                x_win_t = self.t_conv3d(x[:, :, win, :, :, :])
                x_win_s = self.s_conv3d(x[:, :, win, :, :, :])
            if self.use_dcn:
                if self.only_dcn is False:
                    x_win_t = x_win_t + self.t_dcn(x[:, :, win, :, :, :])
                    x_win_s = x_win_s + self.s_dcn(x[:, :, win, :, :, :])
                else:
                    x_win_t = self.t_dcn(x[:, :, win, :, :, :])
                    x_win_s = self.s_dcn(x[:, :, win, :, :, :])
            x_win_t = self.t_norm_act(x_win_t)
            x_win_s = self.s_norm_act(x_win_s)
            x_win_t = self.max_pool(x_win_t).squeeze(2) + self.avg_pool(x_win_t).squeeze(2)
            x_win_s = self.max_pool(x_win_s).squeeze(2) + self.avg_pool(x_win_s).squeeze(2)
            x_m[:, :, win, :, :] = torch.cat([x_win_t, x_win_s], dim=1)
        out = self.m_conv3d(x_m)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, mid_channel, n_class, filter_cfg=None, head_num=[2, 1]):
        super(Decoder, self).__init__()
        self.fpn_in = nn.ModuleList()
        self.fpn_refine = nn.ModuleList()
        self.fpn_in_norm = nn.ModuleList()
        for i in range(len(in_channels)):
            self.fpn_in_norm.append(
                nn.GroupNorm(in_channels[i] // 4, in_channels[i]),
            )
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv3d(in_channels[i], mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.GroupNorm(mid_channel // 4, mid_channel),
                    nn.SiLU(inplace=True),
                )
            )
        for i in range(len(in_channels) - 1):
            self.fpn_refine.append(
                nn.Sequential(
                    nn.Conv3d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.GroupNorm(mid_channel // 4, mid_channel),
                    nn.SiLU(inplace=True),
                )
            )
        self.head_list = nn.ModuleList()
        self.head_list.append(nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'))
        for _ in range(head_num[0]):
            self.head_list.append(
                nn.Sequential(
                    nn.Conv3d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.GroupNorm(mid_channel // 4, mid_channel),
                    nn.SiLU(inplace=True),
                )
            )
        self.head_list.append(nn.Conv3d(in_channels=mid_channel, out_channels=n_class, kernel_size=head_num[1], stride=1, padding=(head_num[1] - 1) // 2, bias=True))
        if filter_cfg is not None:
            self.radius = filter_cfg['filter_radius']
            self.filter_type = filter_cfg['filter_type']
            self.scale = filter_cfg['scale']
            self.use_filter = True
        else:
            self.use_filter = False

    def forward(self, x):
        out = []
        for i in range(len(self.fpn_in)):
            if self.use_filter:
                if self.filter_type == 'pre':
                    out.append(self.fpn_in[i](self._fourier_filter(self.fpn_in_norm[i](x[i]), self.radius[i], self.scale)))
                else:
                    TypeError
            else:
                out.append(self.fpn_in[i](self.fpn_in_norm[i](x[i])))

        for i in range(len(self.fpn_refine), 0, -1):
            out[i - 1] = self.fpn_refine[i - 1](out[i - 1] + F.interpolate(out[i], out[i - 1].size()[-3:], mode='nearest'))
        out = out[0]
        for head in self.head_list:
            out = head(out)
        return out

    @staticmethod
    def _fourier_filter(x, radius, scale):
        if radius <= 0.:
            return x
        x_freq = fft.fftshift(fft.fftn(x, dim=(-2, -1), norm='ortho'), dim=(-2, -1))
        B, C, F, R, A = x_freq.shape
        crow, ccol = R // 2, A // 2
        mask = torch.ones((B, C, F, R, A), device=x.device, dtype=x_freq.real.dtype)
        mask[..., crow - radius: crow + radius, ccol - radius: ccol + radius] = scale
        x_freq = x_freq * mask
        x_freq = fft.ifftn(ifftshift(x_freq, dim=(-2, -1)), dim=(-2, -1), norm='ortho').real
        return x_freq


class D2FANet(nn.Module):
    def __init__(self, mnet_cfg=(4, 32), n_class=3, depths=(2, 2, 4, 2), channels=(64, 128, 256, 512), drop_rate=0.1, mnet_type=['simDecoupledMnet_avg', True, True], tokenMixer=['c', 'c', 'c', 'c'], fpn_channel=128, filter_cfg=dict(filter_radius=[1, 1, 1, 1], filter_type='pre', scale=0.25), head_num=[1, 1]):
        super(D2FANet, self).__init__()
        if mnet_type[0] == 'mnet':
            self.mnet = MNet(in_chirps=mnet_cfg[0], out_channels=mnet_cfg[1], use_dcn=mnet_type[1])
        elif mnet_type[0] == 'ssfm':
            self.mnet = SSFM(in_chirps=mnet_cfg[0], out_channels=mnet_cfg[1])
        elif mnet_type[0] == 'simDecoupledMnet_max':
            self.mnet = simDecoupledMnet_max(in_chirps=mnet_cfg[0], out_channels=mnet_cfg[1], use_dcn=mnet_type[1], only_dcn=mnet_type[2])
        elif mnet_type[0] == 'simDecoupledMnet_avg':
            self.mnet = simDecoupledMnet_avg(mnet_cfg[0], out_channels=mnet_cfg[1], use_dcn=mnet_type[1], only_dcn=mnet_type[2])
        elif mnet_type[0] == 'simDecoupledMnet_mix':
            self.mnet = simDecoupledMnet_mix(mnet_cfg[0], out_channels=mnet_cfg[1], use_dcn=mnet_type[1], only_dcn=mnet_type[2])
        elif mnet_type[0] == 'fc_avg':
            self.mnet = fcAvg(mnet_cfg[0], out_channels=mnet_cfg[1])
        elif mnet_type[0] == 'dfc_avg':
            self.mnet = dfcAvg(mnet_cfg[0], out_channels=mnet_cfg[1])
        else:
            raise TypeError
        self.contextBranch = ContextBranch(mnet_cfg[1], depths, channels, drop_rate, tokenMixer=tokenMixer)
        self.decoder = Decoder(channels, fpn_channel, n_class, filter_cfg=filter_cfg, head_num=head_num)

    def forward(self, x):
        x_context = self.mnet(x)
        f1, f2, f3, f4 = self.contextBranch(x_context)
        out = self.decoder((f1, f2, f3, f4))
        return out