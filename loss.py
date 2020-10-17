"""
損失関数を定義する
"""
import torch
from torch.nn import functional as F

from model import (
    Discriminator
)

def d_lsgan_loss(
    discriminator: Discriminator,
    trues: torch.Tensor,
    fakes: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    **kwargs
) -> tuple:
    """平均二乗誤差(MSELoss)に基づくDiscriminatorの損失を計算する.
    
    識別器に正解データ(訓練画像)からなるバッチと,
    生成器からの偽データからなるバッチをそれぞれ与える.
    それらの出力からMSELossを計算し, 損失の平均を取る.

    Args:
        discriminator (Discriminator): 識別器モデル
        trues (torch.Tensor): 正解データ(訓練画像)からなるバッチ.
        fakes (torch.Tensor): 生成器が生成した偽データからなるバッチ.
        labels (torch.Tensor): 分類に用いるラベルのバッチ.
        alpha (float): Style Mixingの割合.

    Returns:
        torch.Tensor: 損失(スカラ値).
    """
    d_trues = discriminator.forward(trues, labels, alpha)
    d_fakes = discriminator.forward(fakes, labels, alpha)
    
    loss_trues = F.mse_loss(d_trues, torch.ones_like(d_trues))
    loss_fakes = F.mse_loss(d_fakes, torch.zeros_like(d_fakes))
    
    loss = (loss_trues + loss_fakes) / 2
    
    return (loss, )

def g_lsgan_loss(
    discriminator: Discriminator,
    fakes: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    **kwargs
) -> tuple:
    d_fakes = discriminator.forward(fakes, labels, alpha)
    
    loss = F.mse_loss( d_fakes, torch.ones_like(d_fakes) )
    loss /= 2
    
    return (loss, )

def d_wgan_loss(
    discriminator: Discriminator,
    trues: torch.Tensor,
    fakes: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    **kwargs
) -> tuple:
    epsilon_drift = 1e-3
    lambda_gp = 10
    
    batch_size = fakes.shape[0]
    d_trues = discriminator.forward(trues, labels, alpha)
    d_fakes = discriminator.forward(fakes, labels, alpha)
    
    loss_wd = d_trues.mean() - d_fakes.mean()
    
    # gradient penalty
    epsilon = torch.rand(
        batch_size,
        1,
        1,
        1,
        dtype=fakes.dtype,
        device=fakes.device
    ) # -> shape : [batch_size, 1, 1, 1]
    intpl = epsilon * fakes + (1 - epsilon) * trues
    intpl.requires_grad_()
    
    f = discriminator.forward(intpl, labels, alpha)
    grad = torch.autograd.grad(f.sum(), intpl, create_graph=True)[0]
    grad_norm = grad.reshape(batch_size, -1).norm(dim=1)
    loss_gp = lambda_gp * ((grad_norm - 1) ** 2).mean()
    
    # drift
    loss_drift = epsilon_drift * (d_trues ** 2).mean()
    
    loss = -loss_wd + loss_gp + loss_drift
    wd = loss_wd.item()
    
    return (loss, wd)

def g_wgan_loss(
    discriminator: Discriminator,
    fakes: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    **kwargs
) -> tuple:
    d_fakes = discriminator.forward( fakes, labels, alpha )
    loss = -d_fakes.mean()
    return (loss, )

def d_logistic_loss(
    discriminator: Discriminator,
    trues: torch.Tensor,
    fakes: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    **kwargs # rlgamma
) -> tuple:
    if 'rlgamma' in kwargs:
        rlgamma = kwargs['rlgamma']
    else:
        rlgamma = 10

    d_fakes = discriminator.forward( fakes, labels, alpha )
    trues.requires_grad_()
    d_trues = discriminator.forward( trues, labels, alpha )
    loss = F.softplus(d_fakes).mean() + F.softplus(-d_trues).mean()
    
    if rlgamma > 0:
        grad = torch.autograd.grad(
            d_trues.sum(),
            trues,
            create_graph=True
        )[0]
        loss += rlgamma / 2 * (grad ** 2).sum(dim=(1, 2, 3)).mean()
    
    return (loss, )

def g_logistic_loss(
    discriminator: Discriminator,
    fakes: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    **kwargs
) -> tuple:
    d_fakes = discriminator.forward(fakes, labels, alpha)
    return (F.softplus(-d_fakes).mean(), )