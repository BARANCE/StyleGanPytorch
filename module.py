"""
Style GANモデルのModuleを定義する
Moduleはモデルを構成するいくつかの機能がまとまったレイヤのことを指す
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from layer import (
    PixelNormalization,
    WSConv2d,
    Blur3x3,
    AdaIN,
    NoiseLayer
)

class LatentTransformation(nn.Module):
    """Mapping Netoworkを定義するレイヤ
    (WS付きFC Layerを1x1Convで代用しているらしい)
    """
    def __init__( self, settings=None, label_size=1 ):
        """コンストラクタ
        Mapping Networkを構築する

        Args:
            settings (dict, optional): ハイパーパラメータを定義する辞書. Defaults to None.
            label_size (int, optional): ラベルの分類の数.
              use_labelsがTrueの場合に使われる. Defaults to 1.
        """
        super().__init__()
        
        # Default value
        self.z_dim = 256
        self.w_dim = 256
        self.normalize_latents = True
        self.use_labels = False
        if settings is not None:
            self.z_dim = settings['z_dim']
            self.w_dim = settings['w_dim']
            self.normalize_latents = settings['normalize_latents']
            self.use_labels = settings['use_labels']
        
        self.latent_normalization = PixelNormalization(settings) if self.normalize_latents else None
        activation = nn.LeakyReLU(negative_slope=0.2)
        
        self.latent_transform = nn.Sequential(
            WSConv2d( self.z_dim * 2 if self.use_labels else self.z_dim, self.z_dim, 1, 1, 0 ),
            activation,
            WSConv2d(self.z_dim, self.z_dim, 1, 1, 0),
            activation,
            WSConv2d(self.z_dim, self.z_dim, 1, 1, 0),
            activation,
            WSConv2d(self.z_dim, self.z_dim, 1, 1, 0),
            activation,
            WSConv2d(self.z_dim, self.z_dim, 1, 1, 0),
            activation,
            WSConv2d(self.z_dim, self.z_dim, 1, 1, 0),
            activation,
            WSConv2d(self.z_dim, self.w_dim, 1, 1, 0),
            activation
        )
        
        if self.use_labels:
            self.label_embed = nn.Embedding( label_size, self.z_dim )
        else:
            self.label_embed = None

    def forward( self, latent, labels ):
        latent = latent.reshape(-1, self.z_dim, 1, 1)
        
        if self.label_embed is not None:
            labels = self.label_embed(labels)
            if labels.shape[1] >= 2:
                # label_sizeが2以上だと動作しないので削減してみる(根拠なし)
                labels = labels.sum(dim=1)
                labels = labels.reshape((labels.shape[0], 1, labels.shape[1]))
            labels = labels.reshape(-1, self.z_dim, 1, 1)
            latent = torch.cat([latent, labels], dim=1)
        
        if self.latent_normalization is not None:
            latent = self.latent_normalization(latent)
        
        out = self.latent_transform(latent)
        return out

class SynthFirstBlock(nn.Module):
    """Synthesis Networkのうち, 4x4の画像を生成する最初のブロックを表すレイヤ
    (最初のUpsample処理の手前までのブロック)
    下記のレイヤを含む
    - Synthesis Networkの入力となる固定テンソル
    - Noise Layer x2
    - AdaIN Layer x2 (前レイヤの出力とStyle画像wの2つのテンソルを入力とする)
    - WSConv Layer (3x3) x1
    """
    def __init__(
        self,
        start_dim: torch.int32,
        output_dim: torch.int32,
        w_dim: torch.int32,
        base_image_init: str,
        use_noise: torch.bool
    ):
        """コンストラクタ
        Synthesis Networkの最初のブロックを定義する.

        Args:
            start_dim (torch.int32): Synthesis Nwtowrkの入力となる固定テンソルのチャンネル数
            output_dim (torch.int32): このブロックの出力となるテンソルのチャンネル数
            w_dim (torch.int32): スタイル画像wのチャンネル数
            base_image_init (str): 入力テンソルの作成方法を指定.
              'zeros', 'ones', 'zero_normal', 'one_normal'から選択する.
            use_noise (torch.bool): ノイズ加算を行うかどうかのbool値.
              Trueの場合はNoiseを付加する. Falseの場合はNoiseを付加しない.
        """
        super().__init__()
        
        # Synthesis networkの入力画像
        base_image = torch.empty(1, start_dim, 4, 4) # [1, C, 4, 4]
        self.base_image = nn.Parameter(base_image)
        
        if base_image_init == 'zeros':
            nn.init.zeros_(self.base_image)
        elif base_image_init == 'ones':
            nn.init.ones_(self.base_image)
        elif base_image_init == 'zero_normal':
            nn.init.normal_(self.base_image, 0, 1) # mean=0, std=1
        elif base_image_init == 'one_normal':
            nn.init.normal_(self.base_image, 1, 1) # mean=1, std=1
        else:
            print(f"Invalid base_image_init: {base_image_init}")
            exit(1)
        
        # WSConv Layer (3x3 kernel)
        self.conv = WSConv2d(
            start_dim, output_dim, kernel_size=3, stride=1, padding=1
        )
        
        self.noise1 = NoiseLayer(start_dim, 4)
        self.noise2 = NoiseLayer(output_dim, 4)
        if not use_noise:
            # noise加算割合をゼロに設定し、伝搬時にノイズ計算を行わないようにする
            self.noise1.noise_scale.zeros_()
            self.noise1.fixed = True
            self.noise2.noise_scale.zeros_()
            self.noise2.fixed = True
        
        self.adain1 = AdaIN(start_dim, w_dim)
        self.adain2 = AdaIN(output_dim, w_dim)
        
        self.activation = nn.LeakyReLU(negative_slope=0.2)
    
    def forward( self, w1:torch.Tensor, w2:torch.Tensor ) -> torch.Tensor:
        """順方向伝搬
        Synthesis Networkの最初のブロックの処理.
        このブロックの入力テンソルはモデル内で生成されるため, 入力はスタイル画像のみ.

        Args:
            w1 (torch.Tensor): スタイル画像(1)
            w2 (torch.Tensor): スタイル画像(2)

        Returns:
            torch.Tensor: このブロックの出力. 続いてSynthesis Networkの残りのブロックに入れる.
        """
        batch_size = w1.shape[0]
        
        # 固定テンソル(base_image)をバッチサイズ分複製する
        x = self.base_image.expand(batch_size, -1, -1, -1)
        
        # Noise付加(1回目)
        x = self.noise1(x)
        x = self.activation(x)
        
        # AdaINでスタイル画像を合流させる(1回目)
        x = self.adain1(x, w1)
        
        # Weight Scale付きConvolution
        x = self.conv(x)
        
        # Noise付加(2回目)
        x = self.noise2(x)
        x = self.activation(x)
        
        # AdaINでスタイル画像を合流させる(2回目)
        x = self.adain2(x, w2)
        
        return x

class SynthBlock(nn.Module):
    def __init__(
        self,
        input_dim: torch.int32,
        output_dim: torch.int32,
        output_size: torch.int32,
        w_dim: torch.int32,
        upsample_mode: str,
        use_blur: torch.bool,
        use_noise: torch.bool
    ):
        """コンストラクタ
        Synthesis Networkの最初以外のブロック1個を定義する.
        ブロックを1個通過するごとに解像度が1段階アップする.
        (PGGANに基づいたモデルのため, 解像度アップは段階的に行われる)

        Args:
            input_dim (torch.int32): このブロックの入力テンソルのチャンネル数
            output_dim (torch.int32): このブロックの出力テンソルのチャンネル数
            output_size (torch.int32): このブロックの出力テンソルのサイズ(横幅/縦幅)
            w_dim (torch.int32): スタイル(潜在状態)wのチャンネル数
            upsample_mode (str): Synthesis NetworkのUpsampeレイヤの動作を指定する.
              torch.nn.functional.interpolate()のmodeとして,
              'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'の中から指定.
            use_blur (torch.bool): Blur処理の有無を表すbool値.
              Trueの場合はBlurを行う. Falseの場合はBlurを行わない.
            use_noise (torch.bool): ノイズ加算を行うかどうかのbool値.
              Trueの場合はNoiseを付加する. Falseの場合はNoiseを付加しない.
        """
        super().__init__()
        
        # Weight Scaling付きConvolution層(x2)
        self.conv1 = WSConv2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv2 = WSConv2d(
            in_channels=output_dim,
            out_channels=output_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Blue3x3
        if use_blur:
            self.blur = Blur3x3()
        else:
            self.blur = None
        
        # Noise Block
        self.noise1 = NoiseLayer(output_dim, output_size)
        self.noise2 = NoiseLayer(output_dim, output_size)
        if not use_noise:
            # noise加算割合をゼロに設定し、伝搬時にノイズ計算を行わないようにする
            self.noise1.noise_scale.zeros_()
            self.noise1.fixed = True
            self.noise2.noise_scale.zeros_()
            self.noise2.fixed = True
        
        # AdaIN Block
        self.adain1 = AdaIN(output_dim, w_dim)
        self.adain2 = AdaIN(output_dim, w_dim)
        
        # Activation (Leaky ReLU)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        
        # Upsample mode
        self.upsample_mode = upsample_mode
    
    def forward(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor
    ) -> torch.Tensor:
        """順方向伝搬
        Synthesis Networkの最初以外のブロック1個分の処理
        このブロックを複数個組み合わせてSynthesis Networkを構築する.
        このブロックの入力は, このブロックがt番目のブロックだとした場合,
        t-1番目のSynthesis Blockが出力したテンソルとなる.

        Args:
            x (torch.Tensor): このレイヤの入力テンソル.
              これは, 前のSynthesis Blockが出力したテンソルである.
            w1 (torch.Tensor): スタイル(潜在状態)を表すテンソル(1個目)
            w2 (torch.Tensor): スタイル(潜在状態)を表すテンソル(2個目)

        Returns:
            torch.Tensor: このブロックの出力.
        """
        # Upsample Block
        x = F.interpolate(
            input=x,
            scale_factor=2,
            mode=self.upsample_mode,
            align_corners=False # Test
        )
        
        # Weight Scaling付きConvolution (1回目)
        x = self.conv1(x)
        
        # Blur3x3
        if self.blur is not None:
            x = self.blur(x)
        
        # Noise付加 (1回目)
        x = self.noise1(x)
        x = self.activation(x)
        
        # AdaINでスタイル(潜在状態)を合流させる
        x = self.adain1(x, w1)
        
        # Weight Scaling付きConvolution (2回目)
        x = self.conv2(x)
        
        # Noise付加 (2回目)
        x = self.noise2(x)
        x = self.activation(x)
        
        # AdaINでスタイル(潜在状態)を合流させる
        x = self.adain2(x, w2)
        
        return x

class SynthesisModule(nn.Module):
    """GeneratorのSynthesis Networkを表すレイヤ
    """
    def __init__( self, settings=None ):
        """コンストラクタ
        Synthesis Networkを構築する.
        (GeneratorはMapping NetworkとSynthesis Networkからなる)
        モデルの設定は, コンストラクタに渡すsettings辞書で指定する.
        - w_dim (torch.int32): スタイル(潜在状態)のチャンネル数
        - upsample_mode (str): Synthesis NetworkのUpsampeレイヤの動作を指定する.
          torch.nn.functional.interpolate()のmodeとして,
          'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'の中から指定.
        - use_blur (torch.bool): Blur処理の有無を表すbool値.
          Trueの場合はBlurを行う. Falseの場合はBlurを行わない.
        - use_noise (torch.bool): ノイズ加算を行うかどうかのbool値.
          Trueの場合はNoiseを付加する. Falseの場合はNoiseを付加しない.
        - base_image_init (str): 入力テンソルの作成方法を指定.
          'zeros', 'ones', 'zero_normal', 'one_normal'から選択する.
        
        Args:
            settings (dict, optional): ハイパーパラメータを定義する辞書. Defaults to None.
              本レイヤに対して有効な設定値は上記を参照.
        """
        super().__init__()
        
        # Default settings
        self.w_dim = 256
        self.upsample_mode = 'bilinear'
        self.use_blur = False
        self.use_noise = True
        self.base_image_init = 'one_normal'
        if settings is not None:
            self.w_dim = settings['w_dim']
            self.upsample_mode = settings['upsample_mode']
            self.use_blur = settings['use_blur']
            self.use_noise = settings['use_noise']
            self.base_image_init = settings['base_image_init']
        
        # Synthesis Block List
        self.blocks = nn.ModuleList([
            SynthFirstBlock(
                start_dim=256,
                output_dim=256,
                w_dim=self.w_dim,
                base_image_init=self.base_image_init,
                use_noise=self.use_noise
            ), # --> output size : 4x4
            SynthBlock(
                input_dim=256,
                output_dim=256,
                output_size=8,
                w_dim=self.w_dim,
                upsample_mode=self.upsample_mode,
                use_blur=self.use_blur,
                use_noise=self.use_noise
            ), # --> output size : 8x8
            SynthBlock(
                input_dim=256,
                output_dim=256,
                output_size=16,
                w_dim=self.w_dim,
                upsample_mode=self.upsample_mode,
                use_blur=self.use_blur,
                use_noise=self.use_noise
            ), # --> output size : 16x16
            SynthBlock(
                input_dim=256,
                output_dim=128,
                output_size=32,
                w_dim=self.w_dim,
                upsample_mode=self.upsample_mode,
                use_blur=self.use_blur,
                use_noise=self.use_noise
            ), # --> output size : 32x32
            SynthBlock(
                input_dim=128,
                output_dim=64,
                output_size=64,
                w_dim=self.w_dim,
                upsample_mode=self.upsample_mode,
                use_blur=self.use_blur,
                use_noise=self.use_noise
            ), # --> output size : 64x64
            SynthBlock(
                input_dim=64,
                output_dim=32,
                output_size=128,
                w_dim=self.w_dim,
                upsample_mode=self.upsample_mode,
                use_blur=self.use_blur,
                use_noise=self.use_noise
            ), # --> output size : 128x128
            SynthBlock(
                input_dim=32,
                output_dim=16,
                output_size=256,
                w_dim=self.w_dim,
                upsample_mode=self.upsample_mode,
                use_blur=self.use_blur,
                use_noise=self.use_noise
            ), # --> output size : 256x256
        ])
        
        # to RGBS
        self.to_rgbs = nn.ModuleList([
            WSConv2d(
                in_channels=256,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0,
                gain=1
            ),
            WSConv2d(
                in_channels=256,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0,
                gain=1
            ),
            WSConv2d(
                in_channels=256,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0,
                gain=1
            ),
            WSConv2d(
                in_channels=128,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0,
                gain=1
            ),
            WSConv2d(
                in_channels=64,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0,
                gain=1
            ),
            WSConv2d(
                in_channels=32,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0,
                gain=1
            ),
            WSConv2d(
                in_channels=16,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0,
                gain=1
            ),
        ])
        
        # 解像度アップのレベル(初期値:1)
        self.register_buffer('level', torch.tensor(1, dtype=torch.int32))
    
    def set_noise_fixed( self, fixed: torch.bool ):
        """NoiseLayerの生成するnoiseを固定するかどうかを変更する.

        Args:
            fixed (torch.bool): noise固定するかどうかをbool値で選択.
              Trueの場合はNoiseを固定する. Falseの場合は固定しない.
        """
        for module in self.modules():
            if isinstance(module, NoiseLayer):
                module.fixed = fixed
    
    def forward(
        self,
        w: torch.Tensor,
        alpha=1.
    ) -> torch.Tensor:

        # w is [batch_size, level*2, w_dim, 1, 1]
        
        # 解像度アップのレベル(初期値:1)
        level = self.level.item()
        
        # Synthesis First Block
        w1 = w[:, 0]
        w2 = w[:, 1]
        x = self.blocks[0](w1, w2)
        
        if level == 1:
            # levelが1の場合は, それ以降のSynthesis Networkは使用しない.
            # First Block追加後, 画像に変換する.
            x = self.to_rgbs[0](x)
            return x
        
        # levelが2以上の場合の処理
        for idx_level in range(1, level - 1):
            w1 = w[:, idx_level * 2]
            w2 = w[:, idx_level * 2 + 1]
            x = self.blocks[idx_level](x, w1, w2)
        
        # alphaを使うための処理
        x2 = x
        
        w1 = w[:, (level - 1) * 2]
        w2 = w[:, (level - 1) * 2 + 1]
        x2 = self.blocks[level - 1](x2, w1, w2)
        x2 = self.to_rgbs[level - 1](x2)
        
        if alpha == 1:
            # 最後の層まで通常通り出力.
            x = x2
        else:
            # 最終層と, 最終層の1つ手前の出力を線形補間で混合する.
            # 混合割合はalphaで指定する.
            # alphaが1に近いほどx2, 0に近いほどx1に近い結果となる.
            x1 = self.to_rgbs[level - 2](x)
            x1 = F.interpolate(
                x1,
                scale_factor=2,
                mode=self.upsample_mode,
                align_corners=False
            )
            x = torch.lerp(x1, x2, alpha) # 混合
        
        return x
    
    def write_histogram( self, writer: SummaryWriter, step: torch.int32 ):
        for idx_level in range( self.level.item() ):
            block = self.blocks[idx_level]
            
            # パラメータの名前と実体を全て取り出す
            for name, param in block.named_parameters():
                writer.add_histogram(
                    f"g_synth_block{idx_level}/{name}",
                    param.cpu().data.numpy(),
                    step
                )
        
        for name, param in self.to_rgbs.named_parameters():
            writer.add_histogram(
                f"g_synth_block.torgb/{name}",
                param.cpu().data.numpy(),
                step
            )

class DBlock(nn.Module):
    """Discriminatorを構成するPGGANの最後以外のブロック1個分を表すクラス
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_blur: bool
    ):
        """Discriminatorの最後以外の各ブロック1個を構築する.
        準備するブロックの数はSynthesis Networkに存在するブロックの数と常に同じ.

        Args:
            input_dim (int): このブロックの入力テンソルのチャンネル数
            output_dim (int): このブロックの出力テンソルのチャンネル数
            use_blur (bool): Blur処理の有無を表すbool値.
              Trueの場合はBlurを行う. Falseの場合はBlurを行わない.
        """
        super().__init__()
        
        # Weight Scaling付きConvolution層(x2)
        self.conv1 = WSConv2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv2 = WSConv2d(
            in_channels=output_dim,
            out_channels=output_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Blue3x3
        if use_blur:
            self.blur = Blur3x3()
        else:
            self.blur = None
        
        self.activation = nn.LeakyReLU(negative_slope=0.2)
    
    def forward( self, x: torch.Tensor ) -> torch.Tensor:
        """順方向伝搬
        Discriminatorの最後以外のブロック1個分の処理
        このブロックを複数個組み合わせてDiscriminatorを構築する.

        Args:
            x (torch.Tensor): このレイヤの入力テンソル

        Returns:
            torch.Tensor: このブロックの出力.
        """
        x = self.conv1(x)
        x = self.activation(x)
        if self.blur is not None:
            x = self.blur(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = F.avg_pool2d(x, kernel_size=2)
        return x

class DLastBlock(nn.Module):
    """Discriminatorを構成するPGGANの最後のブロックを表すクラス
    """
    def __init__( self, input_dim: int, label_size=1 ):
        """Discriminatorの最後のブロックを構築する.

        Args:
            input_dim (int): このブロックの入力チャンネルの数
            label_size (int): ラベルの分類の数. 通常は1を指定する.
        """
        super().__init__()
        
        # Weight Scaling付きConvolution層(x3)
        self.conv1 = WSConv2d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv2 = WSConv2d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=4,
            stride=1,
            padding=0
        )
        self.conv3 = WSConv2d(
            in_channels=input_dim,
            out_channels=label_size,
            kernel_size=1,
            stride=1,
            padding=0,
            gain=1
        )
        
        # 活性化関数
        self.activation = nn.LeakyReLU(negative_slope=0.2)
    
    def forward( self, x: torch.Tensor ) -> torch.Tensor:
        """順方向伝搬
        Discriminatorの最終ブロックの処理

        Args:
            x (torch.Tensor): このレイヤの入力テンソル
              このテンソルのサイズ(縦x横)は4x4である必要がある.

        Returns:
            torch.Tensor: このブロックの出力.
              入力テンソルの画像サイズが4x4である場合, このテンソルの画像サイズは1x1になる.
              また, 出力チャンネル数はlabel_sizeとなる.
        """
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        return x