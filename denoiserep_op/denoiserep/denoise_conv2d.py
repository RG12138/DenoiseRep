import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .denoise_layer import DenoiseLayer


def count_conv2d_layers(model):
    num = 0
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            num += 1
        else:
            num += count_conv2d_layers(module)
    return num


class DenoiseConv2d(DenoiseLayer):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
        T: int,
        timesteps: int,
        weight: int,
        bias: int,
    ):
        DenoiseLayer.__init__(self, input_dim, output_dim, timesteps, hidden_dim=2048)
        self.kernel_size = kernel_size
        self.stride =  stride
        self.padding = padding
        self.dm = nn.Conv2d(input_dim, output_dim, kernel_size, stride = 1, padding = kernel_size[0] // 2)
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
        new_weight = self.weight - self._ct_1(self.T) * F.conv2d(self.weight, self.dm.weight, self.dm.bias, stride=self.dm.stride, padding=self.dm.padding)
        return new_weight

    def fuse_bias(self):
        new_bias = None
        if self.bias is not None:
            new_bias = self._ct_2(self.T) * self._ct_3(self.T) * self.bias
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
        return  F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding) 


def convert_denoise_conv2d(model: nn.Module, layer_num: int, timesteps: int = 1000):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and (module.kernel_size[0] % 2 == 1):
            setattr(
                model,
                name,
                DenoiseConv2d(
                    input_dim=module.in_channels,
                    output_dim=module.in_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    T=layer_num,
                    timesteps=timesteps,
                    weight=module.weight.data.clone(),
                    bias=module.bias.data.clone() if module.bias is not None else None,
                ),
            )
            layer_num -= 1
        else:
            convert_denoise_conv2d(module, layer_num, timesteps)
    return model
