# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from the Brevitas CIFAR-10 CNV example
# (Xilinx/brevitas: src/brevitas_examples/bnn_pynq/models/CNV.py)
# for use in this thesis project.


import torch
import torch.nn as nn
from dependencies import value

from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import FloatToIntImplType, RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.quant.solver import ActQuantSolver, WeightQuantSolver
from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear


class CommonQuant(ExtendedInjector):
    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_per_output_channel = False
    narrow_range = True
    signed = True

    @value
    def quant_type(bit_width):
        if bit_width is None:
            return QuantType.FP
        elif bit_width == 1:
            return QuantType.BINARY
        else:
            return QuantType.INT


class CommonWeightQuant(CommonQuant, WeightQuantSolver):
    scaling_const = 1.0


class CommonActQuant(CommonQuant, ActQuantSolver):
    min_val = -1.0
    max_val = 1.0


class TensorNorm(nn.Module):
    def __init__(self, eps=1e-4, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var", torch.ones(1))

    def forward(self, x):
        if self.training:
            mean = x.mean()
            unbias_var = x.var(unbiased=True)
            biased_var = x.var(unbiased=False)

            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.detach())
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * unbias_var.detach())

            inv_std = 1.0 / torch.sqrt(biased_var + self.eps)
            return (x - mean) * inv_std * self.weight + self.bias
        else:
            return ((x - self.running_mean) / torch.sqrt(self.running_var + self.eps)) * self.weight + self.bias


class CIFAR10Model(nn.Module):
    """
    Brevitas-style CNV for CIFAR-10.
    Default structure follows the standard CNV example:
      conv_channels = (64, 64, 128, 128, 256, 256)
      fc_features   = (512, 512)
    Pooling is applied after conv blocks 2 and 4 by default.
    Input images are expected in [0, 1] from torchvision.ToTensor().
    The forward maps them to [-1, 1] via x = 2*x - 1.
    """
    def __init__(
        self, n_classes=10, in_ch=3,
        weight_bits=2, act_bits=2, in_bits=8,
        conv_channels=(64, 64, 128, 128, 256, 256),
        fc_features=(512, 512),
        pool_after=(False, True, False, True, False, False),
        kernel_size=3,
        img_h=32, img_w=32,
    ):
        super().__init__()

        assert len(conv_channels) == len(pool_after)

        conv_layers = []
        linear_layers = []

        conv_layers += [
            QuantIdentity(
                act_quant=CommonActQuant,
                bit_width=in_bits,
                min_val=-1.0,
                max_val=1.0 - 2.0 ** (-7),
                narrow_range=False,
                restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            )
        ]

        prev_ch = in_ch
        for out_ch, do_pool in zip(conv_channels, pool_after):
            conv_layers += [
                QuantConv2d(
                    in_channels=prev_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bits,
                ),
                nn.BatchNorm2d(out_ch, eps=1e-4),
                QuantIdentity(
                    act_quant=CommonActQuant,
                    bit_width=act_bits,
                ),
            ]
            if do_pool:
                conv_layers += [nn.MaxPool2d(kernel_size=2)]
            prev_ch = out_ch

        self.conv_features = nn.Sequential(*conv_layers)
        self.flatten = nn.Flatten()

        # Use eval mode temporarily for the shape-inference pass
        self.conv_features.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, img_h, img_w)
            y = self.flatten(self.conv_features(dummy))
            flat_features = y.shape[1]
        self.conv_features.train()

        prev_features = flat_features
        for out_features in fc_features:
            linear_layers += [
                QuantLinear(
                    in_features=prev_features,
                    out_features=out_features,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bits,
                ),
                nn.BatchNorm1d(out_features, eps=1e-4),
                QuantIdentity(
                    act_quant=CommonActQuant,
                    bit_width=act_bits,
                ),
            ]
            prev_features = out_features

        linear_layers += [
            QuantLinear(
                in_features=prev_features,
                out_features=n_classes,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bits,
            ),
            TensorNorm(),
        ]

        self.linear_features = nn.Sequential(*linear_layers)

        for m in self.modules():
            if isinstance(m, (QuantConv2d, QuantLinear)):
                torch.nn.init.uniform_(m.weight.data, -1.0, 1.0)


    def clip_weights(self, min_val=-1.0, max_val=1.0):
        for m in self.modules():
            if isinstance(m, (QuantConv2d, QuantLinear)):
                m.weight.data.clamp_(min_val, max_val)


    def forward(self, x):
        x = 2.0 * x - 1.0
        x = self.conv_features(x)
        x = self.flatten(x)
        x = self.linear_features(x)
        return x