import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .denoise_layer import DenoiseLayer


def count_linear_layers(model):
    num = 0
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            num += 1
        else:
            num += count_linear_layers(module)
    return num


def count_vit_linear_layers(model):
    num = 0
    for name, module in model.named_children():
        if "mlp" in name.lower():
            for sub_name, sub_module in module.named_children():
                if isinstance(sub_module, nn.Linear):
                    num += 1
        else:
            num += count_vit_linear_layers(module)
    return num


class DenoiseLinear(DenoiseLayer):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        T: int,
        timesteps: int,
        weight: int,
        bias: int,
    ):
        DenoiseLayer.__init__(self, input_dim, output_dim, timesteps, hidden_dim=2048)
        self.dm = nn.Linear(input_dim, output_dim)
        self.T = torch.tensor(T)
        self.weight = nn.Parameter(weight)
        if bias == None or bias.shape[0] == 0:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias)
        self.ploss = 0

    def _ct_1(self, t):
        return (1 - self.alphas[t]) / (
            self.sqrt_one_minus_alphas_cumprod[t] * self.sqrt_alphas[t]
        )

    def _ct_2(self, t):
        return self.sqrt_betas[t]

    def _ct_3(self, t):
        return 1.0 / self.sqrt_alphas[t]

    def fuse_weight(self):
        def T(w):
            return w.transpose(0, 1)

        new_weight = T(
            T(self.weight) * self._ct_3(self.T)
            - self._ct_1(self.T) * T(self.dm.weight) @ T(self.weight)
        )
        return new_weight

    def fuse_bias(self):
        def T(w):
            return w.transpose(0, 1)

        new_bias = self._ct_2(self.T) * torch.randn(
            self.weight.shape[1], device=self.weight.device
        ) @ T(self.weight) - self._ct_1(self.T) * self.dm.bias @ T(self.weight)
        if self.bias != None:
            new_bias += self.bias
        return new_bias

    def fuse_parameters(self):
        self.weight, self.bias = nn.Parameter(self.fuse_weight()), nn.Parameter(
            self.fuse_bias()
        )

    def get_ploss(self):
        loss = self.ploss
        self.ploss = 0
        return loss

    def forward(self, x, t=None):
        if t is None:
            t = torch.full((x.shape[0],), self.T, device=x.device, dtype=torch.long)
        weight, bias = self.weight, self.bias
        if self.training:
            x_noise, p_loss = self.p_losses(x, t)
            self.ploss += p_loss
            if self.dm.weight.requires_grad == False:
                weight = self.fuse_weight()
                bias = self.fuse_bias()
        return F.linear(x, weight, bias)


def convert_denoise_linear(model: nn.Module, layer_num: int, timesteps: int = 1000):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(
                model,
                name,
                DenoiseLinear(
                    input_dim=module.in_features,
                    output_dim=module.in_features,
                    T=layer_num,
                    timesteps=timesteps,
                    weight=module.weight.data.clone(),
                    bias=module.bias.data.clone() if module.bias is not None else None,
                ),
            )
            layer_num -= 1
        else:
            convert_denoise_linear(module, layer_num, timesteps)
    return model


def convert_vit_denoise_linear(model: nn.Module, layer_num: int, timesteps: int = 1000):
    for name, module in model.named_children():
        if "mlp" in name.lower():
            for sub_name, sub_module in module.named_children():
                if isinstance(sub_module, nn.Linear):
                    setattr(
                        module,
                        sub_name,
                        DenoiseLinear(
                            input_dim=sub_module.in_features,
                            output_dim=sub_module.in_features,
                            T=layer_num,
                            timesteps=timesteps,
                            weight=sub_module.weight.data.clone(),
                            bias=sub_module.bias.data.clone()
                            if sub_module.bias is not None
                            else None,
                        ),
                    )
                    layer_num -= 1
        else:
            convert_vit_denoise_linear(module, layer_num, timesteps)
    return model
