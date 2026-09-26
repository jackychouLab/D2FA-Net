import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from thop import profile
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from config.selector import rudet_configs, MDRSSM_configs
from model.backbone.ChirpFeatureExtractor import MNet

# =========================================================================
# === Mamba 依赖 ===
# =========================================================================
try:
    from mamba_ssm import Mamba
except ImportError:
    print("=" * 50)
    print("错误：未找到 'mamba_ssm' 包。")
    print("请先安装 Mamba: pip install mamba-ssm")
    print("=" * 50)
    raise


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PositionalEncoding3D(nn.Module):
    # (保留，尽管在 Mamba 逻辑中可能未被积极使用，但为保持代码结构完整性)
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(tensor.type())
        emb[:, :, :, :self.channels] = emb_x
        emb[:, :, :, self.channels:2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels:] = emb_z

        return emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)


class BiMama(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
        self.d_model = d_model
        # 允许传递其它超参数
        mamba_kwargs = dict(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, **kwargs)
        self.mamba_fwd = Mamba(**mamba_kwargs)
        self.mamba_bwd = Mamba(**mamba_kwargs)
        self.merge_layer = nn.Linear(d_model, d_model)

    def forward(self, x):
        # 1. 正向
        out_fwd = self.mamba_fwd(x)
        # 2. 反向
        x_rev = torch.flip(x, dims=[1])
        out_bwd_rev = self.mamba_bwd(x_rev)
        out_bwd = torch.flip(out_bwd_rev, dims=[1])
        out_merge = out_fwd + out_bwd
        x_bidir = self.merge_layer(out_merge)
        return x_bidir


class MambaBlock3D(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False, use_BiMamba=True, d_state=16, d_conv=4, expand=2):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        if use_BiMamba:
            self.mamba = BiMama(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """

        B, D, H, W, C = x.shape
        shortcut = x
        x_norm = self.norm1(x)
        # Flatten 3D space to 1D sequence
        x_flat = x_norm.view(B, -1, C)  # (B, D*H*W, C)
        # Apply Mamba
        if self.use_checkpoint:
            x_mamba = checkpoint.checkpoint(self.mamba, x_flat)
        else:
            x_mamba = self.mamba(x_flat)
        # Reshape back to 3D
        x_mamba_3d = x_mamba.view(B, D, H, W, C)
        # First residual connection
        x = shortcut + self.drop_path(x_mamba_3d)
        # Second residual connection (MLP)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer (Kept as-is)
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm, last=False):
        super().__init__()
        self.last = last
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.conv = nn.Conv3d(in_channels=dim, out_channels=dim * 2, kernel_size=(3, 3, 3), stride=(1, 2, 2),
                              padding=(1, 1, 1))
        self.bn = nn.BatchNorm3d(num_features=dim * 2)
        self.relu = nn.ReLU()
        self.conv2a = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(9, 5, 5), stride=(1, 1, 1),
                                padding=(4, 2, 2))
        self.conv2b = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(9, 5, 5), stride=(2, 2, 2),
                                padding=(4, 2, 2))
        self.gn2a = nn.GroupNorm(num_groups=32, num_channels=128)
        self.gn2b = nn.GroupNorm(num_groups=32, num_channels=128)
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.bn2b = nn.BatchNorm3d(num_features=128)
        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(9, 5, 5), stride=(1, 1, 1),
                                padding=(4, 2, 2))
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(9, 5, 5), stride=(1, 2, 2),
                                padding=(4, 2, 2))
        self.gn3a = nn.GroupNorm(num_groups=32, num_channels=256)
        self.gn3b = nn.GroupNorm(num_groups=32, num_channels=256)
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.bn3b = nn.BatchNorm3d(num_features=256)

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        if self.last:
            x = self.relu(self.gn3a(self.conv3a(x)))
            x = self.relu(self.gn3b(self.conv3b(x)))
        else:
            x = self.relu(self.gn2a(self.conv2a(x)))
            x = self.relu(self.gn2b(self.conv2b(x)))
        x = x.permute(0, 2, 3, 4, 1)
        return x

class PatchExpand_Up(nn.Module):
    # (Kept as-is)
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm, last=False):
        super().__init__()
        self.last = last
        self.input_resolution = input_resolution
        self.dim_scale = dim_scale
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

        self.convt = nn.ConvTranspose3d(in_channels=dim, out_channels=int(dim / 2), kernel_size=(4, 6, 6),
                                        stride=(2, 2, 2), padding=(1, 2, 2))
        self.convt2 = nn.ConvTranspose3d(in_channels=dim, out_channels=3, kernel_size=(4, 6, 6), stride=(2, 2, 2),
                                         padding=(1, 2, 2))
        self.prelu = nn.PReLU()
        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)

    def forward(self, x):
        if self.last:
            x = self.upsample2(x)
        else:
            x = self.upsample(x)
        return x

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim_scale = dim_scale
        self.dim = dim
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        self.conv = nn.Conv3d(in_channels=self.dim, out_channels=int(self.dim / 2), kernel_size=(1, 1, 1))  # 调整通道进行上采样
        self.gn = nn.GroupNorm(num_groups=int(self.dim / 8), num_channels=int(self.dim / 2))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.gn(self.conv(x)))
        x = self.upsample(x)
        return x

class BasicLayer_up(nn.Module):
    def __init__(self, dim, input_resolution, depth, mlp_ratio=4., drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList()
        for i in range(depth):
            current_drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path
            self.blocks.append(
                MambaBlock3D(
                    dim=self.dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=current_drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint
                )
            )

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand_Up(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None
        self.upsamplelast = PatchExpand_Up(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer, last=True)
        self.conv = nn.Conv3d(in_channels=dim, out_channels=int(dim / 2), kernel_size=1)
        self.gn = nn.GroupNorm(num_groups=int(dim / 8), num_channels=int(dim / 2))
        self.relu = nn.ReLU()


    def forward(self, x):
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b d h w c')
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
        x = x.permute(0, 4, 1, 2, 3)
        if self.upsample is not None:
            x = self.relu(self.gn(self.conv(x)))
            x = self.upsample(x)
        else:
            x = self.upsamplelast(x)
        return x

class DAMLayerConcat(nn.Module):
    def __init__(self, in_channel):
        super(DAMLayerConcat, self).__init__()

        self.maina = nn.Conv3d(in_channels=int(in_channel), out_channels=int(in_channel / 4), kernel_size=(1, 1, 1),
                               stride=(1, 1, 1), padding=(0, 0, 0))
        self.mainb = nn.Conv3d(in_channels=int(in_channel / 4), out_channels=int(in_channel / 4), kernel_size=(3, 1, 1),
                               stride=(1, 1, 1), padding=(1, 0, 0))
        self.branch2 = nn.Conv3d(in_channels=int(in_channel / 4), out_channels=int(in_channel / 4),
                                 kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0))
        self.branch3 = nn.Conv3d(in_channels=int(in_channel / 4), out_channels=int(in_channel / 4),
                                 kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1))
        self.branch4 = nn.Conv3d(in_channels=int(in_channel / 4), out_channels=int(in_channel / 4),
                                 kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.last = nn.Conv3d(in_channels=int(in_channel * 2), out_channels=int(in_channel), kernel_size=(1, 1, 1),
                              stride=(1, 1, 1), padding=(0, 0, 0))
        self.gnmaina = nn.GroupNorm(num_groups=int(in_channel / 16), num_channels=int(in_channel / 4))
        self.gnmainb = nn.GroupNorm(num_groups=int(in_channel / 16), num_channels=int(in_channel / 4))
        self.gnbranch2 = nn.GroupNorm(num_groups=int(in_channel / 16), num_channels=int(in_channel / 4))
        self.gnbranch3 = nn.GroupNorm(num_groups=int(in_channel / 16), num_channels=int(in_channel / 4))
        self.gnbranch4 = nn.GroupNorm(num_groups=int(in_channel / 16), num_channels=int(in_channel / 4))
        self.gnlast = nn.GroupNorm(num_groups=int(in_channel / 4), num_channels=int(in_channel))
        self.relu = nn.ReLU()

    def forward(self, x):
        branch1 = x  # B 64 8 64 64
        x = self.relu(self.gnmaina(self.maina(branch1)))
        x = self.relu(self.gnmainb(self.mainb(x)))
        branch6 = x  # B 64 8 64 64
        branch2 = self.relu(self.gnbranch2(self.branch2(x)))  # B 16 8 64 64
        branch3 = self.relu(self.gnbranch3(self.branch3(x)))  # B 16 8 64 64
        branch4 = self.relu(self.gnbranch4(self.branch4(x)))  # B 16 8 64 64
        x = torch.cat((branch1, branch2, branch3, branch4, branch6), 1)  # 4 256 8 64 64
        x = self.relu(self.gnlast(self.last(x)))  # B 128 8 64 64
        return x

# =========================================================================
# === NEW: 基于门控自注意力和交叉注意力的特征增强空间 ===
# =========================================================================

class GatedSelfAttention3D(nn.Module):
    """
    Gated Self-Attention (GSA) for 3D features.

    结合了:
    1. 空间门控 (Spatial Gating): 使用深度可分离卷积提取局部上下文作为门控。
    2. 通道注意力 (Channel Selections): 动态调整通道权重。

    目的: 在与另一模态交互前，先净化自身特征 (Self-Refinement)。
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        # 1. 门控生成分支 (Gating Branch)
        # 使用 Depth-wise Conv 提取局部 3D 上下文，轻量级
        self.gate_conv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
            nn.Sigmoid()  # 生成 0~1 的门控系数
        )

        # 2. 特征变换分支 (Feature Branch)
        self.feat_conv = nn.Conv3d(dim, dim, kernel_size=1, bias=False)

        # 3. 融合后的投影
        self.proj = nn.Conv3d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, C, D, H, W)
        res = x

        # 为了使用 LayerNorm，需要调整维度
        # (B, C, D, H, W) -> (B, D, H, W, C)
        x_in = x.permute(0, 2, 3, 4, 1).contiguous()
        x_in = self.norm(x_in)
        # (B, D, H, W, C) -> (B, C, D, H, W)
        x_in = x_in.permute(0, 4, 1, 2, 3).contiguous()

        # 生成门控
        gate = self.gate_conv(x_in)
        # 变换特征
        feat = self.feat_conv(x_in)

        # 门控自注意力操作: Output = Gate * Feature
        out = gate * feat
        out = self.proj(out)
        return res + out


class LinearCrossAttention3D(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.dim = dim

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, x_source, x_target):
        B, C, D, H, W = x_source.shape
        N = D * H * W

        x_s_flat = x_source.view(B, C, N).permute(0, 2, 1)
        x_t_flat = x_target.view(B, C, N).permute(0, 2, 1)

        x_s_flat = self.norm_q(x_s_flat)
        x_t_flat = self.norm_kv(x_t_flat)

        q = self.to_q(x_s_flat)
        kv = self.to_kv(x_t_flat)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-2)

        context = torch.matmul(k.transpose(-2, -1), v)
        out = torch.matmul(q, context)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = out.permute(0, 2, 1).view(B, C, D, H, W)

        return out + x_source  # Residual Connection


class AttentionalFusionSpace(nn.Module):
    def __init__(self, dim, drop=0.):
        super().__init__()
        self.ra_gsa = GatedSelfAttention3D(dim)
        self.adc_gsa = GatedSelfAttention3D(dim)

        self.cross_ra_to_adc = LinearCrossAttention3D(dim, num_heads=4)  # RA attends to ADC
        self.cross_adc_to_ra = LinearCrossAttention3D(dim, num_heads=4)  # ADC attends to RA

        self.fusion_conv = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=max(1, int(dim / 8)), num_channels=dim),
            nn.GELU()
        )

        self.drop = nn.Dropout(drop)

    def forward(self, x_ra, x_adc):
        ra_refined = self.ra_gsa(x_ra)
        adc_refined = self.adc_gsa(x_adc)
        ra_enhanced = self.cross_adc_to_ra(x_source=ra_refined, x_target=adc_refined)
        adc_enhanced = self.cross_ra_to_adc(x_source=adc_refined, x_target=ra_refined)
        x_cat = torch.cat([ra_enhanced, adc_enhanced], dim=1)
        out = self.fusion_conv(x_cat)
        out = self.drop(out)
        return out


#TODO:EDSM消融
class EightWayMambaLayer(nn.Module):
    def __init__(self, in_channel, d_state=16, d_conv=4, expand=2,
                 use_d=True, use_h=True, use_w=True, use_hw=True):  # <--- 增加控制开关
        super(EightWayMambaLayer, self).__init__()

        self.in_channel = in_channel
        self.d_model = in_channel // 4  # Mamba的特征维度

        # 保存开关状态
        self.use_d = use_d
        self.use_h = use_h
        self.use_w = use_w
        self.use_hw = use_hw

        self.maina = nn.Conv3d(in_channels=in_channel, out_channels=self.d_model, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.gnmaina = nn.GroupNorm(num_groups=int(self.d_model / 4), num_channels=self.d_model)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(self.d_model)
        mamba_kwargs = dict(d_model=self.d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        if self.use_d:
            self.mamba_d = BiMama(**mamba_kwargs)
        if self.use_h:
            self.mamba_h = BiMama(**mamba_kwargs)
        if self.use_w:
            self.mamba_w = BiMama(**mamba_kwargs)
        if self.use_hw:
            self.mamba_hw = BiMama(**mamba_kwargs)

        num_active_branches = sum([self.use_d, self.use_h, self.use_w, self.use_hw])
        concat_channels = in_channel + num_active_branches * self.d_model

        self.last = nn.Conv3d(in_channels=concat_channels, out_channels=in_channel,
                              kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.gnlast = nn.GroupNorm(num_groups=int(in_channel / 4), num_channels=in_channel)

    def forward(self, x):
        branch1 = x
        x_in = self.relu(self.gnmaina(self.maina(branch1)))
        B, C_r, D, H, W = x_in.shape
        out_list = [branch1]

        if self.use_d:
            x_d = x_in.permute(0, 3, 4, 2, 1).contiguous()
            x_d_norm = self.norm(x_d)
            x_d_flat = x_d_norm.view(B * H * W, D, C_r)
            out_d = self.mamba_d(x_d_flat)
            branch_d = out_d.view(B, H, W, D, C_r).permute(0, 4, 3, 1, 2).contiguous()
            out_list.append(branch_d)

        if self.use_h:
            x_h = x_in.permute(0, 2, 4, 3, 1).contiguous()
            x_h_norm = self.norm(x_h)
            x_h_flat = x_h_norm.view(B * D * W, H, C_r)
            out_h = self.mamba_h(x_h_flat)
            branch_h = out_h.view(B, D, W, H, C_r).permute(0, 4, 1, 3, 2).contiguous()
            out_list.append(branch_h)

        if self.use_w:
            x_w = x_in.permute(0, 2, 3, 4, 1).contiguous()
            x_w_norm = self.norm(x_w)
            x_w_flat = x_w_norm.view(B * D * H, W, C_r)
            out_w = self.mamba_w(x_w_flat)
            branch_w = out_w.view(B, D, H, W, C_r).permute(0, 4, 1, 2, 3).contiguous()
            out_list.append(branch_w)

        if self.use_hw:
            x_hw = x_in.permute(0, 2, 3, 4, 1).contiguous()
            x_hw_norm = self.norm(x_hw)
            x_hw_flat = x_hw_norm.view(B * D, H * W, C_r)
            out_hw = self.mamba_hw(x_hw_flat)
            branch_hw = out_hw.view(B, D, H, W, C_r).permute(0, 4, 1, 2, 3).contiguous()
            out_list.append(branch_hw)
        x_cat = torch.cat(out_list, dim=1)
        out = self.relu(self.gnlast(self.last(x_cat)))
        return out

class BasicLayer(nn.Module):
    def __init__(self,
                 input_resolution,
                 dim,
                 depth,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 use_BiMamba=False,
                 use_EightWay=False):
        super().__init__()

        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            MambaBlock3D(
                dim=dim,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                use_BiMamba=use_BiMamba
            )
            for i in range(depth)])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

        self.downsamplelast = PatchMerging(dim=dim, norm_layer=norm_layer, last=True)

        self.conv1 = nn.Conv3d(in_channels=dim, out_channels=int(dim / 4), kernel_size=1)
        self.bn1 = nn.BatchNorm3d(int(dim / 4))

        self.conv2 = nn.Conv3d(in_channels=int(dim / 4), out_channels=dim, kernel_size=1)
        self.bn2 = nn.BatchNorm3d(dim)

        self.relu = nn.ReLU()
        if use_EightWay:
            self.DAM = EightWayMambaLayer(dim, use_d=True, use_h=True, use_w=True, use_hw=True)
        else:
            self.DAM = nn.Identity()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # (B,64,8,64,64) --> (B,64,8,64,64)
        x = self.DAM(x)

        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b d h w c')

        for idx, blk in enumerate(self.blocks):
            x = blk(x)

        x = x.reshape(B, D, H, W, -1)

        if self.downsample is not None:
            x_ = x
            x_ = self.norm(x_)
            x_ = rearrange(x_, 'b d h w c -> b c d h w')
            if C != 128:
                x = self.downsample(x)
            else:
                x = self.downsamplelast(x)
            x = rearrange(x, 'b d h w c -> b c d h w')
        else:
            x = self.norm(x)
            x = rearrange(x, 'b d h w c -> b c d h w')
            x_ = x
        return x, x_


class PatchEmbed3D(nn.Module):

    def __init__(self, img_size=(16, 128, 128), patch_size=(4, 4, 4), in_chans=2, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[1] // patch_size[1]]
        self.patches_resolution = patches_resolution  # (4, 32, 32)


        self.relu = nn.ReLU()
        self.gn1a = nn.GroupNorm(num_groups=int(embed_dim / 4), num_channels=embed_dim)
        self.gn1b = nn.GroupNorm(num_groups=int(embed_dim / 4), num_channels=embed_dim)

        self.conv1a = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, kernel_size=(9, 5, 5), stride=(1, 1, 1),
                                padding=(4, 2, 2))
        self.conv1b = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(9, 5, 5), stride=(2, 2, 2),
                                padding=(4, 2, 2))

    def forward(self, x):
        x = self.relu(self.gn1a(self.conv1a(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128)
        x = self.relu(self.gn1b(self.conv1b(x)))  # (B, 64, W, 128, 128) -> (B, 64, W/2, 64, 64)

        return x


class ADCPatchEmbed3D(nn.Module):

    def __init__(self, in_chans=4, embed_dim=64, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.relu = nn.ReLU()
        # (假设 embed_dim=64, 64/4 = 16)
        self.gn1a = nn.GroupNorm(num_groups=int(embed_dim / 4), num_channels=embed_dim)
        self.gn1b = nn.GroupNorm(num_groups=int(embed_dim / 4), num_channels=embed_dim)

        self.conv1a = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, kernel_size=(9, 5, 5), stride=(1, 1, 1),
                                padding=(4, 2, 2))
        self.conv1b = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(9, 5, 5), stride=(2, 2, 4),
                                padding=(4, 2, 2))

        # 暴露 patches_resolution 以便 U-Net 的其余部分使用
        self.patches_resolution = [4, 32, 32]

    def forward(self, x):
        x = self.relu(self.gn1a(self.conv1a(x)))
        x = self.relu(self.gn1b(self.conv1b(x)))
        return x


class ClutterSuppressionExpert(nn.Module):
    def __init__(self, K, in_channels, ref_win_size, guard_win_size):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.epsilon = 1e-6
        if ref_win_size % 2 == 0 or guard_win_size % 2 == 0:
            raise ValueError(f"[Expert K={K}] ref_win_size and guard_win_size must be odd")
        if guard_win_size >= ref_win_size:
            raise ValueError(f"[Expert K={K}] guard_win_size must be smaller than ref_win_size")
        ref_padding = ref_win_size // 2
        guard_padding = guard_win_size // 2
        self.total_pool = nn.AvgPool3d(kernel_size=(1, ref_win_size, ref_win_size),
                                       stride=1,
                                       padding=(0, ref_padding, ref_padding))
        self.guard_pool = nn.AvgPool3d(kernel_size=(1, guard_win_size, guard_win_size),
                                       stride=1,
                                       padding=(0, guard_padding, guard_padding))
        self.ref_area = ref_win_size * ref_win_size
        self.guard_area = guard_win_size * guard_win_size
        self.num_ref_cells = self.ref_area - self.guard_area
        self.alpha_scaler = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.attn_conv1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1)
        self.attn_relu = nn.ReLU()
        self.attn_conv2 = nn.Conv3d(in_channels // 2, 1, kernel_size=1)
        self.attn_sigmoid = nn.Sigmoid()

    def _pca_suppress(self, x):
        B, C, D, H, W = x.shape
        if self.K == 0:
            return x.clone()
        M = x.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
        M_centered = M
        Sigma = M_centered.T @ M_centered
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(Sigma.double(), UPLO='U')
            eigenvectors = eigenvectors.to(x.dtype)
        except torch.linalg.LinAlgError:
            jitter = torch.eye(D, device=x.device, dtype=torch.float64) * 1e-8
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(Sigma.double() + jitter, UPLO='U')
                eigenvectors = eigenvectors.to(x.dtype)
            except Exception as e:
                print(f"Error during PCA eigh even with jitter: {e}. Returning original.")
                return x.clone()
        current_K = min(self.K, D)
        if current_K == 0:
            return x.clone()
        V_c = eigenvectors[:, -current_K:]
        projection_matrix = torch.eye(D, device=x.device, dtype=x.dtype) - (V_c @ V_c.T)
        S = M_centered @ projection_matrix
        x_suppressed = S.view(B, C, H, W, D).permute(0, 1, 4, 2, 3).contiguous()
        return x_suppressed

    def forward(self, x):
        x_suppressed = self._pca_suppress(x)
        x_total_mean = self.total_pool(x_suppressed)
        x_guard_mean = self.guard_pool(x_suppressed)
        x_ref_mean_approx = (x_total_mean * self.ref_area - x_guard_mean * self.guard_area) / (
                self.num_ref_cells + self.epsilon)
        threshold_feature = self.alpha_scaler(x_ref_mean_approx)
        significance = F.relu(x_suppressed - threshold_feature)
        attn_map = self.attn_conv1(significance)
        attn_map = self.attn_relu(attn_map)
        attn_map = self.attn_conv2(attn_map)
        attn_map = self.attn_sigmoid(attn_map)
        x_suppressed_gated = attn_map * x_suppressed
        return x_suppressed_gated


class SubSpaceClutterSuppression(nn.Module):
    def __init__(self, in_channels, expert_configs: list[dict]):
        super().__init__()
        self.num_experts = len(expert_configs)
        self.in_channels = in_channels
        print(f"MoE-LGASF_DFCA: Initializing {self.num_experts} heterogeneous experts...")
        self.gate = nn.Conv3d(in_channels, self.num_experts, kernel_size=1)
        self.experts = nn.ModuleList()
        for i, config in enumerate(expert_configs):
            K = config.get('K', 1)
            ref = config.get('ref_win_size', 7)
            guard = config.get('guard_win_size', 3)
            print(f"  ... Expert {i + 1}: K={K}, RefWin={ref}, GuardWin={guard}")
            self.experts.append(
                ClutterSuppressionExpert(
                    K=K,
                    in_channels=in_channels,
                    ref_win_size=ref,
                    guard_win_size=guard
                )
            )
        self.merge_conv = nn.Conv3d(in_channels * 2, in_channels, kernel_size=1)
        self.merge_relu = nn.ReLU()
        self.latest_gate_weights = None

    def forward(self, x):
        x_original = x
        gate_logits = self.gate(x_original)
        gate_weights = F.softmax(gate_logits, dim=1)

        if not self.training and rudet_configs['viz_SSCS']:
            self.latest_gate_weights = gate_weights.detach().cpu()

        final_suppressed_gated = torch.zeros_like(x_original)

        for i, expert in enumerate(self.experts):
            expert_out = expert(x_original)
            weights = gate_weights[:, i:i + 1, ...]
            final_suppressed_gated += expert_out * weights

        x_concat = torch.cat([x_original, final_suppressed_gated], dim=1)
        x_fused = self.merge_conv(x_concat)
        x_fused = self.merge_relu(x_fused)
        return x_fused


class MambaUNet(nn.Module):
    def __init__(self,
                 img_size=(128, 128, 128),
                 patch_size=(4, 2, 2),
                 in_chans=4,
                 num_classes=3,
                 embed_dim=96,
                 depths=[2, 2, 1],
                 depths_decoder=[1, 2, 2],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 use_mnet=False,
                 use_moe_SSCS=False,
                 clutter_expert_configs=None,
                 use_posEmbed=False,
                 use_BiMamba=False,
                 use_EightWay=False,
                 use_ADCBranch=False,
                 **kwargs):
        super().__init__()

        print("=" * 50)
        print("MamRODNet SubSpaceClutterSuppression Initializing...")
        print(f"... depths:{depths}; depths_decoder:{depths_decoder}; embed_dims:{embed_dim}")
        if use_moe_SSCS:
            print(f"... SubSpaceClutterSuppression with MoE Enabled: Yes ({len(clutter_expert_configs)} experts)")
        else:
            print(f"... SubSpaceClutterSuppression with MoE Enabled: No (Using nn.Identity)")
        if use_posEmbed:
            print(f"... Our Radar Position Embedding Enabled: Yes)")
        else:
            print(f"... Our Radar Position Embedding Enabled: No)")
        if use_BiMamba:
            print(f"... BiMamba for U-shape Enabled: Yes)")
        else:
            print(f"... Mamba for U-shape Enabled: Yes)")
        if use_ADCBranch:
            print(f"... ADC Branch Enabled: Yes)")
        else:
            print(f"... Mamba for U-shape Enabled: No)")

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.embed_dim = embed_dim
        self.use_posEmbed = use_posEmbed
        self.use_adcBranch = use_ADCBranch
        self.use_mnet = use_mnet

        if self.use_mnet:
            in_chans = rudet_configs['mnet_cfg'][1]
        self.patch_embed = PatchEmbed3D(img_size=img_size,
                                        patch_size=patch_size,
                                        in_chans=in_chans,
                                        embed_dim=embed_dim,
                                        norm_layer=norm_layer if self.patch_norm else None)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if use_moe_SSCS:
            self.clutter_supperession = SubSpaceClutterSuppression(
                in_channels=self.embed_dim,
                expert_configs=clutter_expert_configs
            )
        else:
            self.clutter_supperession = nn.Identity()

        if self.use_posEmbed:
            self.pos_embed = PositionalEncoding3D(self.embed_dim)
            self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                  self.patches_resolution[1] // (2 ** i_layer),
                                  self.patches_resolution[2] // (2 ** i_layer)),
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
                use_BiMamba=use_BiMamba,
                use_EightWay=use_EightWay)
            self.layers.append(layer)

        if self.use_adcBranch:
            self.adc_patch_embed = ADCPatchEmbed3D(
                in_chans=2,
                embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None
            )

            if use_posEmbed:
                self.pos_embed = PositionalEncoding3D(self.embed_dim)
                self.pos_drop = nn.Dropout(p=drop_rate)

            adc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
            self.adc_layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(
                    input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                      self.patches_resolution[1] // (2 ** i_layer),
                                      self.patches_resolution[2] // (2 ** i_layer)),
                    dim=int(embed_dim * 2 ** i_layer),
                    depth=depths[i_layer],
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=adc_dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                    use_checkpoint=use_checkpoint,
                    use_BiMamba=use_BiMamba,
                    use_EightWay=False)
                self.adc_layers.append(layer)

            self.skip_fusion_layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                dim = int(embed_dim * 2 ** i_layer)
                in_dim = dim * 2
                out_dim = dim
                self.skip_fusion_layers.append(
                    nn.Sequential(
                        nn.Conv3d(in_dim, out_dim, kernel_size=1, bias=False),
                        nn.GroupNorm(num_groups=int(out_dim / 8), num_channels=out_dim),  # 使用 8 组
                        nn.ReLU()
                    ))

            bottleneck_dim = int(embed_dim * 2 ** (self.num_layers - 1))
            self.bottleneck_fusion = nn.Sequential(
                nn.Conv3d(bottleneck_dim * 2, bottleneck_dim, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=int(bottleneck_dim / 8), num_channels=bottleneck_dim),
                nn.ReLU()
            )

        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Conv3d(in_channels=2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      out_channels=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      kernel_size=(1, 1, 1))

            current_res_factor = 2 ** (self.num_layers - 1 - i_layer)
            current_resolution = (self.patches_resolution[0] // current_res_factor,
                                  self.patches_resolution[1] // current_res_factor,
                                  self.patches_resolution[2] // current_res_factor)

            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=current_resolution,
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=current_resolution,
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                        depths[:(self.num_layers - 1 - i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                )

            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.convlast = nn.Conv3d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=(1, 1, 1))

    def forward_features(self, x_ra, x_adc):
        x_ra = self.patch_embed(x_ra)
        x_ra = self.clutter_supperession(x_ra)

        if self.use_posEmbed:
            x_ra = x_ra + self.pos_embed(x_ra)
            x_ra = self.pos_drop(x_ra)

        if self.use_adcBranch:
            x_adc = self.adc_patch_embed(x_adc)
            if self.use_posEmbed:
                x_adc = self.adc_pos_embed(x_adc)
                x_adc = self.adc_pos_drop(x_adc)

            x_downsample = []
            z_ra = x_ra
            z_adc = x_adc

            for i in range(self.num_layers):
                z_ra, skip_ra = self.layers[i](z_ra)
                z_adc, skip_adc = self.adc_layers[i](z_adc)
                fused_skip = self.skip_fusion_layers[i](torch.cat((skip_ra, skip_adc), dim=1))
                x_downsample.append(fused_skip)

            z = self.bottleneck_fusion(torch.cat((z_ra, z_adc), dim=1))
            return z, x_downsample
        else:
            x_downsample = []
            for i, layer in enumerate(self.layers):
                x_ra, x_ = layer(x_ra)
                x_downsample.append(x_)

            return x_ra, x_downsample

    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[2 - inx]], 1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
        return x

    def forward(self, x_ra, x_adc):
        if self.use_adcBranch:
            z, x_downsample = self.forward_features(x_ra, x_adc)
        else:
            z, x_downsample = self.forward_features(x_ra, x_adc=None)
        x_up = self.forward_up_features(z, x_downsample)
        logits = self.convlast(x_up)
        return logits


class MDRSSM(nn.Module):
    def __init__(self, num_classes=1, zero_head=False, embed_dim=96, win_size=7, FLOPs=False,
                 use_moe_SSCS=False,
                 clutter_expert_configs=None,
                 use_posEmbed=False,
                 use_BiMamba=False,
                 use_EightWay=False,
                 use_ADCBranch=False):

        super(MDRSSM, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.FLOPs = FLOPs
        self.embed_dim = embed_dim
        self.win_size = (win_size, win_size, win_size)  # Mamba ignores this
        self.with_mnet = False
        if 'mnet_cfg' in rudet_configs:
            in_chirps_mnet, out_channels_mnet = rudet_configs['mnet_cfg']
            self.mnet = MNet(in_chirps_mnet, out_channels_mnet, conv_op=nn.Conv3d)
            self.with_mnet = True

        self.mamba_unet = MambaUNet(img_size=(8, 64, 64),
                                              patch_size=(2, 2, 2),
                                              in_chans=2,
                                              num_classes=self.num_classes,
                                              embed_dim=self.embed_dim,
                                              depths=[2, 2, 2],
                                              depths_decoder=[2, 2, 2],
                                              mlp_ratio=4.,
                                              drop_rate=0,
                                              drop_path_rate=0.1,
                                              norm_layer=nn.LayerNorm,
                                              patch_norm=True,
                                              use_checkpoint=False,
                                              use_mnet=self.with_mnet,
                                              use_moe_SSCS=use_moe_SSCS,
                                              clutter_expert_configs=clutter_expert_configs,
                                              use_posEmbed=use_posEmbed,
                                              use_BiMamba=use_BiMamba,
                                              use_EightWay=use_EightWay,
                                              use_ADCBranch=use_ADCBranch)

    def forward(self, x_ra, x_adc=None):
        if self.with_mnet:
            x_ra = self.mnet(x_ra)
        logits = self.mamba_unet(x_ra, x_adc)
        return logits



# =========================================================================
# === 测试脚本 (Modified) ===
# =========================================================================
# from fvcore.nn import FlopCountAnalysis, parameter_count_table
#
# if __name__ == '__main__':
#     """T-RODNet (Mamba Version):
#     FLOPs and Params are now calculated *only* using thop.profile.
#     """
#     from config.selector import MDRSSM_configs
#     # 确保 mamba_ssm 可用
#     try:
#         from mamba_ssm import Mamba
#     except ImportError:
#         print("=" * 50)
#         print("请在运行此脚本前安装 mamba_ssm:")
#         print("pip install mamba-ssm")
#         print("=" * 50)
#         exit()
#
#     if torch.cuda.is_available():
#         device = 'cuda'
#         print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
#     else:
#         device = 'cpu'
#         print(f"Using device: {device}. CUDA (GPU) not available.")
#
#     BATCH_SIZE = 1  # 保持 B=1 以便测试
#     n_class = 1
#
#     # RA (原始) 输入: (B, C_ra, D_ra, H_ra, W_ra)
#     input_tensor_ra = torch.rand((BATCH_SIZE, 2, 16, 128, 128), device=device)
#
#     # ADC (新) 输入: (B, C_adc, D_adc, H_adc, W_adc)
#     input_tensor_adc = torch.rand((BATCH_SIZE, 2, 16, 128, 255), device=device)
#
#
#     # FLOPs=True/False 标志不再影响 forward 的返回值
#     net = MDRSSM(num_classes=n_class, embed_dim=64, win_size=4, FLOPs=False,
#                         use_moe_SSCS=MDRSSM_configs['SSCS'],
#                         clutter_expert_configs=MDRSSM_configs['CLUTTER_EXPERT_CONFIGS'] if
#                         MDRSSM_configs['SSCS'] else None,
#                         use_posEmbed=MDRSSM_configs['posEmbed'],
#                         use_BiMamba=MDRSSM_configs['BiMamba'],
#                         use_EightWay=MDRSSM_configs['EightWay'],
#                         use_ADCBranch=MDRSSM_configs['ADC']).cuda()
#
#     print(net) # 取消注释以查看模型结构
#
#     output = net(input_tensor_ra, input_tensor_adc)
#     print(f"\nInput RA shape:  {input_tensor_ra.shape}")
#     print(f"Input ADC shape: {input_tensor_adc.shape}")
#     print(f"Output shape: {output.shape}")
#
#     # 使用 thop.profile 计算 FLOPs 和参数
#     # 注意：thop 可能无法完全准确计算 Mamba (mamba_ssm) 的自定义 CUDA 操作
#     # 结果可能是一个近似值
#     print("\nCalculating FLOPs and Params using thop.profile...")
#     try:
#         flops, params = profile(net, inputs=(input_tensor_ra, input_tensor_adc,))
#         print(f"FLOPs (thop) = {str(flops / 1e9)} G")
#         print(f"Params (thop) = {str(params / 1e6)} M")
#     except Exception as e:
#         print(f"Could not calculate FLOPs using thop.profile: {e}")
#         print("thop may not support Mamba's custom operations.")
#
#     print("\nModel parameter table (fvcore):")
#     print(parameter_count_table(net))