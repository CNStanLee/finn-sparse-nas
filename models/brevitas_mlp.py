import torch.nn as nn
from brevitas.nn import QuantLinear, QuantReLU, QuantHardTanh
from brevitas.core.quant import QuantType
from brevitas.core.scaling import ScalingImplType


class JetSubstructureModel(nn.Module):
    """
    Dense MLP that mimics LogicNets' quantisation style:
    - BN before each activation
    - Input: QuantHardTanh(bit_width=input_abits, max_val=1.0)
    - Hidden: QuantReLU(bit_width=hidden_abits, max_val=1.61, learned scaling)
    - Output: QuantHardTanh(bit_width=output_abits, max_val=1.33)
    Weights are quantised uniformly with weight_bit_width=w_bits.
    """
    def __init__(
        self, in_features, n_classes, hidden=(64, 32, 32, 32),
        w_bits=2, input_abits=2, hidden_abits=2, output_abits=2,
    ):
        super().__init__()
        layers = []

        # Input BN + input quant (HardTanh)
        layers += [
            nn.BatchNorm1d(in_features),
            QuantHardTanh(
                bit_width=input_abits,
                min_val=-1.0, max_val=1.0,
                quant_type=QuantType.INT,
                scaling_impl_type=ScalingImplType.PARAMETER,
            ),
        ]

        prev = in_features
        for h in hidden:
            # Linear -> BN -> QuantReLU
            layers += [
                QuantLinear(prev, h, bias=False, weight_bit_width=w_bits),
                nn.BatchNorm1d(h),
                QuantReLU(
                    bit_width=hidden_abits,
                    max_val=1.61,
                    quant_type=QuantType.INT,
                    scaling_impl_type=ScalingImplType.PARAMETER,
                ),
            ]
            prev = h

        # Final linear
        layers += [
            QuantLinear(prev, n_classes, bias=True, weight_bit_width=w_bits),
            QuantHardTanh(
                bit_width=output_abits,
                min_val=-1.33, max_val=1.33,
                quant_type=QuantType.INT,
                scaling_impl_type=ScalingImplType.PARAMETER,
            ),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)