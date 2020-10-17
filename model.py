"""
Style GANのネットワークモデルを定義する
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from layer import WSConv2d
from module import (
    LatentTransformation,
    SynthesisModule,
    DBlock,
    DLastBlock
)

class Generator(nn.Module):
    """Style-GANのGenerator
    """
    def __init__( self, settings=None, label_size=1 ):
        """コンストラクタ
        Style-GANのGeneratorを構築する.
        Mapping NetworkとSynthesis Networkから構成される.
        また, TruncationTrickのための下準備も行う.
        (TruncationTrick : 学習時と, 学習完了後の画像の生成時で入力を変化させる手法)

        Args:
            settings (dict, optional): ハイパーパラメータを指定する辞書. Defaults to None.
            label_size (int, optional): ラベルの分類の数.
              use_labelsがTrueの場合に使われる. Defaults to 1.
        """
        super().__init__()
        
        # Default Settings
        self.style_mixing_prob = 0.9
        self.z_dim = 256
        if settings is not None:
            self.style_mixing_prob = settings['style_mixing_prob']
            self.z_dim = settings['z_dim']
        
        # Mapping Network
        self.latent_transform = LatentTransformation(settings, label_size)
        self.synthesis_module = SynthesisModule(settings)
        
        # Truncation trick
        # See: https://qiita.com/t-ae/items/afc969c48450507dc421
        # 学習時と学習後の生成時で入力を変えてやることで、
        # 生成画像の多様性は減るもののクオリティは向上するという手法
        self.register_buffer(
            'w_average',
            torch.zeros(1, self.z_dim, 1, 1)
        )
        self.w_average_beta = 0.995
        self.trunc_w_layers = 8
        self.trunc_w_psi = 0.8
    
    def set_level( self, level: int ):
        """PGGANの解像度アップレベルを変更する.

        Args:
            level (int): 解像度アップのレベル(モデルの初期値は1)
        """
        self.synthesis_module.level.fill_(level)
    
    def forward(
        self,
        z: torch.Tensor,
        labels,
        alpha: float
    ) -> torch.Tensor:
        batch_size = z.shape[0]
        level = self.synthesis_module.level.item()
        
        w = self.latent_transform(z, labels)

        # update w_average
        if self.training:
            self.w_average = torch.lerp(
                w.mean(0, keepdim=True).detach(),
                self.w_average,
                self.w_average_beta
            )
        
        # w becomes [B, level*2, z_dim, 1, 1]
        w = w.reshape([batch_size, 1, -1, 1, 1])
        # レベル方向に拡大(level*2個複製する)
        w = w.expand(-1, level * 2, -1, -1, -1)
        
        # Style mixing
        if self.training and level >= 2:
            z_mix = torch.randn_like(z)
            w_mix = self.latent_transform(z_mix, labels)
            for batch_index in range(batch_size):
                if np.random.uniform(0, 1) < self.style_mixing_prob:
                    cross_point = np.random.randint(1, level * 2)
                    #w[batch_index, cross_point:] = w_mix[batch_index] # Error
                    for idx in range(cross_point, w.shape[1]):
                        w[batch_index, idx] = w_mix[batch_index]
        
        # Truncation trick
        if not self.training:
            leaped = torch.lerp(
                self.w_average,
                w[:, self.trunc_w_layers:],
                self.trunc_w_psi
            )
            for idx in range(0, leaped.shape[1]):
                w[:, self.trunc_w_layers + idx] = leaped[:, idx]
            """ Error
            w[:, self.trunc_w_layers:] = torch.lerp(
                self.w_average,
                w[:, self.trunc_w_layers:],
                self.trunc_w_psi
            )
            """
        
        fakes = self.synthesis_module(w, alpha)
        
        return fakes
    
    def write_histogram( self, writer: SummaryWriter, step: int ):
        for name, param in self.latent_transform.named_parameters():
            writer.add_histogram(
                f"g_lt/{name}",
                param.cpu().data.numpy(),
                step
            )
        self.synthesis_module.write_histogram(writer, step)
        writer.add_histogram(
            'w_average',
            self.w_average.cpu().data.numpy(),
            step
        )

class Discriminator(nn.Module):
    """Style-GANのDiscriminator
    """
    def __init__( self, settings=None, label_size=1 ):
        """コンストラクタ
        Style-GANのDiscriminatorを構築する.
        Discriminatorは, PGGANのものと同様である.

        Args:
            settings (dict, optional): ハイパーパラメータを指定する辞書. Defaults to None.
            label_size (int, optional): ラベルの分類の数. Defaults to 1.
        """
        super().__init__()
        
        # Default Parameters
        self.use_blur = False
        self.downsample_mode = 'bilinear'
        self.use_labels = False
        if settings is not None:
            self.use_blur = settings['use_blur']
            self.downsample_mode = settings['upsample_mode']
            self.use_labels = settings['use_labels']
        
        self.from_rgbs = nn.ModuleList([
            WSConv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            WSConv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            WSConv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            WSConv2d(
                in_channels=3,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            WSConv2d(
                in_channels=3,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            WSConv2d(
                in_channels=3,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            WSConv2d(
                in_channels=3,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0
            )
        ])
        
        # labelの使用有無
        if self.use_labels:
            self.label_size = label_size
        else:
            self.label_size = 1
        
        # Discriminatorを構成するPGGANのmodule
        self.blocks = nn.ModuleList([
            DBlock(
                input_dim=16,
                output_dim=32,
                use_blur=self.use_blur
            ),
            DBlock(
                input_dim=32,
                output_dim=64,
                use_blur=self.use_blur
            ),
            DBlock(
                input_dim=64,
                output_dim=128,
                use_blur=self.use_blur
            ),
            DBlock(
                input_dim=128,
                output_dim=256,
                use_blur=self.use_blur
            ),
            DBlock(
                input_dim=256,
                output_dim=256,
                use_blur=self.use_blur
            ),
            DBlock(
                input_dim=256,
                output_dim=256,
                use_blur=self.use_blur
            ),
            DLastBlock(
                input_dim=256,
                label_size=self.label_size
            )
        ])
        
        # 活性化関数
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        
        # 解像度アップの段階
        self.register_buffer(
            'level',
            torch.tensor(1, dtype=torch.int32)
        )
    
    def set_level( self, level: int ):
        """PGGANの解像度アップレベルを変更する.

        Args:
            level (int): 解像度アップのレベル(モデルの初期値は1)
        """
        self.level.fill_(level)
    
    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        level = self.level.item()
        
        if level == 1:
            # levelが1の場合は, 最後のDBlockのみを用いる
            x = self.from_rgbs[-1](x) # -> [batch_size, 256, 4, 4]
            x = self.activation(x)
            x = self.blocks[-1](x) # -> [batch_size, label_size, 1, 1]
        else:
            # levelがそれ以外の場合の処理
            x2 = self.from_rgbs[-level](x)
            x2 = self.activation(x2)
            x2 = self.blocks[-level](x2)
            
            # Style Mixing
            if alpha == 1:
                # Style Mixingなし
                x = x2
            else:
                x1 = F.interpolate(
                    x,
                    scale_factor=0.5,
                    mode=self.downsample_mode,
                    align_corners=False, # Test
                    recompute_scale_factor=True # Test
                )
                x1 = self.from_rgbs[-level + 1](x1)
                
                x = torch.lerp(x1, x2, alpha)
            
            for idx_level in range(1, level):
                x = self.blocks[-level + idx_level](x)
        
        if self.use_labels:
            x = x.reshape([-1, self.label_size])
            # Pytorchはラベルにlong型を使用する
            # See: https://www.hellocybernetics.tech/entry/2017/10/07/013937#torchTensor%E3%81%AE%E5%9E%8B%E5%A4%89%E6%8F%9B
            y = labels
            gat = torch.gather(x, 1, y)
            return gat
        else:
            return x.reshape(-1, 1)