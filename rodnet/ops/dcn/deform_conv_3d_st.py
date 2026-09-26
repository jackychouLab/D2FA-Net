import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _triple
import tdc_deform_conv_cuda_3d as deform_conv_3d_cuda


class DeformConvFunction3D(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        offset,
        weight,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        im2col_step=64,
    ):
        if input.dim() != 5:
            raise ValueError(f"Expected input [B,C,T,H,W], got {input.shape}.")
        if offset.dim() != 5:
            raise ValueError(f"Expected offset [B,C,T,H,W], got {offset.shape}.")

        ctx.stride = _triple(stride)
        ctx.padding = _triple(padding)
        ctx.dilation = _triple(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        input = input.contiguous()
        offset = offset.contiguous()
        weight = weight.contiguous()
        ctx.save_for_backward(input, offset, weight)
        output = input.new_empty(DeformConvFunction3D._output_size(input,weight, ctx.padding, ctx.dilation, ctx.stride,)
        )

        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]
        if not input.is_cuda:
            raise NotImplementedError("3D deformable convolution only supports CUDA.")
        cur_im2col_step = min(ctx.im2col_step, input.shape[0])
        assert input.shape[0] % cur_im2col_step == 0, (
            f"im2col_step must divide batch size. "
            f"Got batch={input.shape[0]}, im2col_step={cur_im2col_step}."
        )
        deform_conv_3d_cuda.deform_conv_forward_cuda(
            input,
            weight,
            offset,
            output,
            ctx.bufs_[0],
            ctx.bufs_[1],
            weight.size(4),  # kW
            weight.size(3),  # kH
            weight.size(2),  # kT
            ctx.stride[2],   # dW
            ctx.stride[1],   # dH
            ctx.stride[0],   # dT
            ctx.padding[2],  # padW
            ctx.padding[1],  # padH
            ctx.padding[0],  # padT
            ctx.dilation[2], # dilationW
            ctx.dilation[1], # dilationH
            ctx.dilation[0], # dilationT
            ctx.groups,
            ctx.deformable_groups,
            cur_im2col_step,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors

        grad_input = grad_offset = grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError("3D deformable convolution only supports CUDA.")

        grad_output = grad_output.contiguous()

        cur_im2col_step = min(ctx.im2col_step, input.shape[0])
        assert input.shape[0] % cur_im2col_step == 0, (
            f"im2col_step must divide batch size. "
            f"Got batch={input.shape[0]}, im2col_step={cur_im2col_step}."
        )

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_input = torch.zeros_like(input)
            grad_offset = torch.zeros_like(offset)
            deform_conv_3d_cuda.deform_conv_backward_input_cuda(
                input,
                offset,
                grad_output,
                grad_input,
                grad_offset,
                weight,
                ctx.bufs_[0],
                weight.size(4),
                weight.size(3),
                weight.size(2),
                ctx.stride[2],
                ctx.stride[1],
                ctx.stride[0],
                ctx.padding[2],
                ctx.padding[1],
                ctx.padding[0],
                ctx.dilation[2],
                ctx.dilation[1],
                ctx.dilation[0],
                ctx.groups,
                ctx.deformable_groups,
                cur_im2col_step,
            )

        if ctx.needs_input_grad[2]:
            grad_weight = torch.zeros_like(weight)

            deform_conv_3d_cuda.deform_conv_backward_parameters_cuda(
                input,
                offset,
                grad_output,
                grad_weight,
                ctx.bufs_[0],
                ctx.bufs_[1],
                weight.size(4),
                weight.size(3),
                weight.size(2),
                ctx.stride[2],
                ctx.stride[1],
                ctx.stride[0],
                ctx.padding[2],
                ctx.padding[1],
                ctx.padding[0],
                ctx.dilation[2],
                ctx.dilation[1],
                ctx.dilation[0],
                ctx.groups,
                ctx.deformable_groups,
                1,
                cur_im2col_step,
            )

        return (
            grad_input,
            grad_offset,
            grad_weight,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        output_size = (input.size(0), weight.size(0))
        for d in range(3):
            in_size = input.size(d + 2)
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            output_size += ((in_size + 2 * padding[d] - kernel) // stride[d] + 1,)
        if not all(s > 0 for s in output_size):
            raise ValueError("convolution input is too small; output would be " + "x".join(map(str, output_size)))
        return output_size

deform_conv_3d = DeformConvFunction3D.apply

class DeformConv3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=False,
        im2col_step=64,
    ):
        super().__init__()
        if bias:
            raise NotImplementedError("This CUDA implementation does not support bias.")
        if in_channels % groups != 0:
            raise ValueError(f"in_channels {in_channels} must be divisible by groups {groups}.")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels {out_channels} must be divisible by groups {groups}.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        nn.init.uniform_(self.weight, -stdv, stdv)

    @property
    def offset_channels(self):
        kt, kh, kw = self.kernel_size
        return self.deformable_groups * 3 * kt * kh * kw

    def _check_offset(self, offset):
        if offset.dim() != 5:
            raise ValueError(f"Expected offset [B,C,T,H,W], got {offset.shape}.")
        if offset.size(1) != self.offset_channels:
            raise ValueError(f"Invalid offset channels: got {offset.size(1)}, "f"expected {self.offset_channels}.")

    def _pad_input_if_needed(self, x):
        pad_t = max(self.kernel_size[0] - x.size(2), 0)
        pad_h = max(self.kernel_size[1] - x.size(3), 0)
        pad_w = max(self.kernel_size[2] - x.size(4), 0)
        if pad_t == 0 and pad_h == 0 and pad_w == 0:
            return x, (0, 0, 0)

        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="constant", value=0)
        return x.contiguous(), (pad_t, pad_h, pad_w)

    @staticmethod
    def _crop_output(out, pads):
        pad_t, pad_h, pad_w = pads
        if pad_t > 0:
            out = out[:, :, :-pad_t, :, :]
        if pad_h > 0:
            out = out[:, :, :, :-pad_h, :]
        if pad_w > 0:
            out = out[:, :, :, :, :-pad_w]
        return out.contiguous()

    def forward(self, x, offset):
        if x.dim() != 5:
            raise ValueError(f"Expected x [B,C,T,H,W], got {x.shape}.")
        self._check_offset(offset)
        x, pads = self._pad_input_if_needed(x)
        out = deform_conv_3d(
            x,
            offset.contiguous(),
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
            self.im2col_step,
        )
        return self._crop_output(out, pads)


def pack_temporal_offset(offset_t, kernel_size, deformable_groups):
    kt, kh, kw = _triple(kernel_size)
    k_total = kt * kh * kw
    b, c, t, h, w = offset_t.shape
    expected = deformable_groups * k_total
    if c != expected:
        raise ValueError(f"Temporal offset channels: got {c}, expected {expected}.")
    offset = offset_t.new_zeros(b, deformable_groups * 3 * k_total, t, h, w)
    for g in range(deformable_groups):
        src = g * k_total
        dst = g * 3 * k_total
        offset[:, dst : dst + 3 * k_total : 3] = offset_t[:, src : src + k_total]
    return offset.contiguous()


def pack_spatial_offset(offset_hw, kernel_size, deformable_groups):
    kt, kh, kw = _triple(kernel_size)
    k_total = kt * kh * kw
    b, c, t, h, w = offset_hw.shape
    expected = deformable_groups * 2 * k_total
    if c != expected:
        raise ValueError(f"Spatial offset channels: got {c}, expected {expected}.")
    offset = offset_hw.new_zeros(b, deformable_groups * 3 * k_total, t, h, w)
    for g in range(deformable_groups):
        src = g * 2 * k_total
        dst = g * 3 * k_total
        offset[:, dst + 1 : dst + 3 * k_total : 3] = offset_hw[:, src : src + 2 * k_total : 2]
        offset[:, dst + 2 : dst + 3 * k_total : 3] = offset_hw[:, src + 1 : src + 2 * k_total : 2]
    return offset.contiguous()


class TemporalDeformConvPack3D(DeformConv3D):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=False,
        im2col_step=64,
    ):
        kt = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        stride_t = stride[0] if isinstance(stride, tuple) else stride
        dilation_t = dilation[0] if isinstance(dilation, tuple) else dilation

        ksize = (kt, 1, 1)
        stride_ = (stride_t, 1, 1)
        dilation_ = (dilation_t, 1, 1)
        padding_ = (kt // 2, 0, 0) if padding is None else (
            padding[0] if isinstance(padding, tuple) else padding,
            0,
            0,
        )
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            groups=groups,
            deformable_groups=deformable_groups,
            bias=bias,
            im2col_step=im2col_step,
        )
        k_total = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        self.conv_offset_t = nn.Conv3d(
            self.in_channels,
            self.deformable_groups * k_total,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        self.init_offset()

    def init_offset(self):
        nn.init.constant_(self.conv_offset_t.weight, 0.0)
        nn.init.constant_(self.conv_offset_t.bias, 0.0)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"Expected x [B,C,T,H,W], got {x.shape}.")
        x, pads = self._pad_input_if_needed(x)
        offset_t = self.conv_offset_t(x)
        # offset_t = torch.tanh(offset_t) * 0.5
        offset = pack_temporal_offset(offset_t, self.kernel_size, self.deformable_groups)
        out = deform_conv_3d(
            x,
            offset,
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
            self.im2col_step,
        )
        return self._crop_output(out, pads)


class SpatialDeformConvPack3D(DeformConv3D):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=False,
        im2col_step=64,
    ):
        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size[-2], kernel_size[-1]
        else:
            kh = kw = kernel_size
        if isinstance(stride, tuple):
            stride_h, stride_w = stride[-2], stride[-1]
        else:
            stride_h = stride_w = stride
        if isinstance(dilation, tuple):
            dilation_h, dilation_w = dilation[-2], dilation[-1]
        else:
            dilation_h = dilation_w = dilation
        ksize = (1, kh, kw)
        stride_ = (1, stride_h, stride_w)
        dilation_ = (1, dilation_h, dilation_w)
        if padding is None:
            padding_ = (0, kh // 2, kw // 2)
        elif isinstance(padding, tuple):
            padding_ = (0, padding[-2], padding[-1])
        else:
            padding_ = (0, padding, padding)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            groups=groups,
            deformable_groups=deformable_groups,
            bias=bias,
            im2col_step=im2col_step,
        )
        k_total = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        self.conv_offset_hw = nn.Conv3d(
            self.in_channels,
            self.deformable_groups * 2 * k_total,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        self.init_offset()

    def init_offset(self):
        nn.init.constant_(self.conv_offset_hw.weight, 0.0)
        nn.init.constant_(self.conv_offset_hw.bias, 0.0)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"Expected x [B,C,T,H,W], got {x.shape}.")
        x, pads = self._pad_input_if_needed(x)
        offset_hw = self.conv_offset_hw(x)
        # offset_hw = torch.tanh(offset_hw)
        offset = pack_spatial_offset(offset_hw, self.kernel_size, self.deformable_groups)
        out = deform_conv_3d(
            x,
            offset,
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
            self.im2col_step,
        )
        return self._crop_output(out, pads)


class DeformConvPack3D_True(DeformConv3D):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=False,
        im2col_step=64,
        offset_scale=1.0,
        use_tanh_offset=False,
    ):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        dilation = _triple(dilation)

        if padding is None:
            padding = tuple(k // 2 for k in kernel_size)
        else:
            padding = _triple(padding)

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deformable_groups=deformable_groups,
            bias=bias,
            im2col_step=im2col_step,
        )

        self.offset_scale = offset_scale
        self.use_tanh_offset = use_tanh_offset

        self.conv_offset = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.offset_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )

        self.init_offset()

    def init_offset(self):
        nn.init.constant_(self.conv_offset.weight, 0.0)
        nn.init.constant_(self.conv_offset.bias, 0.0)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"Expected x [B,C,T,H,W], got {x.shape}.")

        x, pads = self._pad_input_if_needed(x)

        offset = self.conv_offset(x)

        if self.use_tanh_offset:
            offset = torch.tanh(offset) * self.offset_scale
        elif self.offset_scale != 1.0:
            offset = offset * self.offset_scale

        out = deform_conv_3d(
            x,
            offset.contiguous(),
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
            self.im2col_step,
        )

        return self._crop_output(out, pads)