import torch
import torch.nn as nn
import torch.nn.functional as F

# 线性层量化
# 组件
def activation_quant_linear(x, nbit=8, q_group_size=-1):
    """Per token quantization to 8bits. No grouping is needed for quantization

    Args:
        x (Tensor): _description_

    Returns:
        _type_: _description_
    """
    maxx = 2**(nbit-1)-1

    org_x_shape = x.shape
    if q_group_size > 0:
        assert org_x_shape[-1] % q_group_size == 0
        x = x.reshape(-1, q_group_size)

    scale =  x.abs().max(dim=-1, keepdim=True).values.clamp_(min=2e-3) / (maxx-1.0)
    y = (x / scale).round().clamp_(-maxx+0.0, maxx-1.0) * scale

    if q_group_size > 0:
        y = y.reshape(org_x_shape)
    return y

def pseudo_quantize_tensor_linear(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w

# Override
class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, x_nbit=8, w_nbit=8, q_group_size=-1, qmethod='absmean'):
        super(QLinear, self).__init__(in_features, out_features, bias)

        self.x_nbit = x_nbit
        self.w_nbit = w_nbit
        self.qmethod = qmethod
        self.q_group_size = q_group_size

    def forward(self, x):
        w = self.weight
        if self.w_nbit > 0:
            w = w + (pseudo_quantize_tensor_linear(w, self.w_nbit, q_group_size=self.q_group_size ) - w).detach()
        if self.x_nbit > 0:
            x = activation_quant_linear(x, self.x_nbit,q_group_size=self.q_group_size)
        output = F.linear(x, w, self.bias)
        return output

# Conv2d量化
# Activation quantization function
def activation_quant_conv(x, nbit=8, q_group_size=-1):
    maxx = 2**(nbit-1)-1

    org_x_shape = x.shape
    if q_group_size > 0:
        assert org_x_shape[-1] % q_group_size == 0
        x = x.reshape(-1, q_group_size)

    scale =  x.abs().max(dim=-1, keepdim=True).values.clamp_(min=2e-3) / (maxx-1.0)
    y = (x / scale).round().clamp_(-maxx+0.0, maxx-1.0) * scale

    if q_group_size > 0:
        y = y.reshape(org_x_shape)
    return y

# Weight quantization function
def pseudo_quantize_tensor_conv(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w

# Custom quantized Conv2d layer
class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, x_nbit=8, w_nbit=8, q_group_size=-1):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                      padding, dilation, groups, bias)
        self.x_nbit = x_nbit
        self.w_nbit = w_nbit
        self.q_group_size = q_group_size

    def forward(self, x):
        w = self.weight
        if self.w_nbit > 0:
            # Quantize the weights
            w = w + (pseudo_quantize_tensor_conv(w.view(w.size(0), -1), self.w_nbit, q_group_size=self.q_group_size).view_as(w) - w).detach()
        if self.x_nbit > 0:
            # Quantize the input activations
            x = activation_quant_conv(x, self.x_nbit, q_group_size=self.q_group_size)
        # Perform convolution operation
        output = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


