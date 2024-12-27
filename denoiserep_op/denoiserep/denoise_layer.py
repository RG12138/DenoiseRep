import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def _linear_beta_schedule(
    timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.002
):
    return torch.linspace(beta_start, beta_end, timesteps)


def _extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class DenoiseLayer(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, timesteps: int, hidden_dim: int
    ):
        super(DenoiseLayer, self).__init__()
        self.timesteps = timesteps
        betas = _linear_beta_schedule(timesteps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_betas", torch.sqrt(betas))
        self.register_buffer("alphas", alphas)
        self.register_buffer("sqrt_alphas", torch.sqrt(self.alphas))
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer(
            "alphas_cumprod_prev",
            torch.cat((torch.tensor([1.0]), self.alphas_cumprod[:-1])),
        )
        assert self.alphas_cumprod_prev.shape == (self.timesteps,)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - self.alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / self.alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / self.alphas_cumprod - 1)
        )
        self.register_buffer(
            "posterior_variance",
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod),
        )
        d_model = hidden_dim
        time_dim = output_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = _extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, noise=None, loss_type="l1 + l2"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.pred_noise(x_noisy, t)
        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "l1 + l2":
            loss = F.l1_loss(noise, predicted_noise) + F.mse_loss(
                noise, predicted_noise
            )
        else:
            raise NotImplementedError(
                "The specified loss type is not supported. Please choose from 'l1', 'l2', or 'l1 + l2'."
            )

        return x_noisy, loss

    def pred_noise(self, f, t):
        t = self.time_mlp(t)
        if t.shape != f.shape:
            t = t.unsqueeze(1).expand(-1, f.shape[1], -1)
        x = f + t
        # 1x fc
        output = self.dm(x.detach())

        # 2x fc
        # x = self.fc1(x)
        # output = self.fc2(x)
        return output


def fuse_parameters(model):
    for name, module in model.named_children():
        if issubclass(type(module), DenoiseLayer):
            module.fuse_parameters()
        else:
            fuse_parameters(module)
    return model


def get_ploss(model):
    loss = 0
    for name, module in model.named_children():
        if issubclass(type(module), DenoiseLayer):
            loss += module.get_ploss()
        else:
            loss += get_ploss(module)
    return loss


def freeze_denoise_layers(model):
    for name, param in model.named_parameters():
        if "dm" in name or "time_mlp" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def unfreeze_denoise_layers(model):
    for name, param in model.named_parameters():
        if "dm" in name or "time_mlp" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
