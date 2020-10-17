"""
Style GANモデルで使用するレイヤを定義する.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class PixelNormalization(nn.Module):
    """Pixel normalization
    See: https://tzmi.hatenablog.com/entry/2020/05/07/230232
    各入力に対し、チャネル成分の二乗和の平方根で割り算する。
    Normalizationの一種。
    BatchNormalizationのように値の保存は行わない。
    """
    def __init__( self, settings=None ):
        """コンストラクタ
        ハイパーパラメータ(epsilon)の設定
        dictが設定されない場合はデフォルトの値(1e-7)を使用する

        Args:
            settings (dict, optional): ハイパーパラメータ. Defaults to None.
        """
        super().__init__()
        self.epsilon = 1e-7 # デフォルトのepsilon
        if settings is not None:
            self.epsilon = settings['epsilon']
    
    def forward( self, x:torch.Tensor ) -> torch.Tensor:
        """順方向伝搬
        Pixel Normalizationを適用する

        Args:
            x (torch.Tensor): このレイヤの入力Tensor

        Returns:
            torch.Tensor: Pixel Normalizationを適用したこのレイヤの出力Tensor.
        """
        # x is [B, C, H, W]
        x2 = x ** 2
        mean = torch.mean( x2, dim=1, keepdims=True )
        return x * ( torch.rsqrt(mean) + self.epsilon )

class MinibatchStdConcatLayer(nn.Module):
    def __init__( self, settings=None ):
        """コンストラクタ
        settingsで指定するパラメータは下記
        - num_concat (torch.int32): 結合するテンソルの個数
        - group_size (torch.int32): ここで指定した値ごとに計算を行う. バッチサイズを割り切る値であることが必要.
        - use_variance (torch.bool): ?
        - epsilon (torch.float32): 分散を計算する際に分母に加える微小な値.

        Args:
            settings (dict, optional): ハイパーパラメータ. Defaults to None.
        """
        super().__init__()
        
        # default settings
        self.num_concat = 2 # 結合するstd画像の数[0, 2]
        self.group_size = 5
        self.use_variance = False
        self.epsilon = 1e-7
    
    def forward( self, x:torch.Tensor ) -> torch.Tensor:
        """順方向伝搬
        ミニバッチ標準偏差の値からなる画像を入力テンソルに結合する
        (PGGANとは異なり、結合されるテンソルは1～2個)
        結合されるテンソルの数は、settingsにより指定する

        Args:
            x (torch.Tensor): このレイヤの入力Tensor

        Returns:
            torch.Tensor: ミニバッチ標準偏差の計算結果を結合した出力Tensor
        """
        if self.num_concat == 0:
            return x
        
        group_size = self.group_size
        # x is [B, C, H, W]
        B, C, H, W = x.shape
        assert(B % group_size == 0)
        M = B // group_size
        
        x32 = x.to(torch.float32)
        
        # num_concat == 1のケース
        y = x32.reshape( group_size, M, -1 ) # [group_size, M, -1]
        mean = y.mean(0, keepdims=True) # [1, M, -1]
        y = ((y - mean) ** 2).mean(0) # [M, -1]
        if not self.use_variance:
            y = (y + self.epsilon).sqrt()
        y = y.mean(1) # [M] ミニバッチ標準偏差の値
        
        # 値をコピーして入力データと同じサイズの1チャネル画像を生成
        y1 = y.expand([B, 1, H, W])
        y1 = y1.to(x.dtype)
        
        if self.num_concat == 1:
            return torch.cat([x, y1], 1)
        
        # num_concat == 2のケース
        y = x32.reshape( M, group_size, -1 ) # [M, group_size, -1]
        mean = y.mean(1, keepdims=True) # [M, 1, -1]
        y = ((y - mean) ** 2).mean(1) # [M, -1]
        if self.use_variance:
            y = (y + self.epsilon).sqrt()
        y = y.mean(1, keepdims=True) # [M, 1]
        y = y.repeat(1, group_size) # [M, group_size]
        y = y.reshape(-1, 1, 1, 1) # [B, 1, 1, 1]
        y2 = y.expand([B, 1, H, W]) # [B, 1, H, W]
        y2 = y2.to(x.dtype)
        
        return torch.cat([x, y1, y2], 1) # 2個の画像を結合

class Blur3x3(nn.Module):
    def __init__( self ):
        """コンストラクタ
        forward時にConv2Dで使用するフィルタを定義する
        """
        super().__init__()
        
        f = np.array([1, 2, 1], dtype=np.float32)
        f = torch.from_numpy(f)
        f = f[None, :] * f[:, None] # 3x1 * 1*3 = 3x3
        f /= f.sum()
        f = f.reshape([1, 1, 3, 3])
        
        # register_buffer()を使用してモデルにフィルタ用の値を保存しておく
        self.register_buffer('filter', f)
        
    def forward( self, x:torch.Tensor ) -> torch.Tensor:
        """順方向伝搬
        Blur3x3フィルタを適用する
        内部ではConvolution Layerが使用されている.
        このフィルタを通した前後でテンソルのサイズは変わらない.

        Args:
            x (torch.Tensor): このレイヤの入力Tensor

        Returns:
            torch.Tensor: Blur3x3を適用したテンソル
        """
        ch = x.shape[1] # チャンネル数
        ft = self.filter # 保存しておいたフィルタ
        ft = ft.expand(ch, -1, -1, -1) # チャンネル数と合うように拡大
        out = F.conv2d(x, ft, padding=1, groups=ch) # ConvLayer. pad=1なのでサイズは変わらない
        return out

class WSConv2d(nn.Module):
    """Weight Scale付きConvolution Layer
    """
    def __init__(
        self,
        in_channels:torch.int32,
        out_channels:torch.int32,
        kernel_size:torch.int32,
        stride:torch.int32,
        padding:torch.int32,
        gain=np.sqrt(2)
    ):
        """コンストラクタ
        WeightScale制御を付加した畳み込みレイヤを定義する
        このレイヤでは、畳み込み演算で用いられる重み(フィルタ)を、scale倍して適用する
        scaleの値は、ハイパーパラメータgainの値により決定される

        Args:
            in_channels (torch.int32): 入力データのチャンネル数
            out_channels (torch.int32): 出力データのチャンネル数
            kernel_size (torch.int32): カーネルサイズ
            stride (torch.int32): ストライド
            padding (torch.int32): パディング
            gain (torch.float32, optional): ハイパーパラメータ. Defaults to np.sqrt(2).
        """
        super().__init__()
        
        # weight(filter)の初期値
        weight = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        init.normal_(weight)
        self.weight = nn.Parameter(weight)
        
        # weightに乗算するscaleの値を決定(weight自身を更新するわけではない)
        scale = gain / np.sqrt(in_channels * kernel_size * kernel_size)
        scale = torch.tensor(scale, dtype=torch.float32)
        self.register_buffer('scale', scale)
        
        # biasの初期値
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        self.stride = stride
        self.padding = padding
    
    def forward( self, x:torch.Tensor ) -> torch.Tensor:
        """順方向伝搬
        Weight Scaleされた重みを利用して畳み込み演算を行う

        Args:
            x (torch.Tensor): このレイヤの入力Tensor

        Returns:
            torch.Tensor: 畳み込み演算後のテンソル
        """
        scaled_weight = self.weight * self.scale
        out = F.conv2d(
            x, scaled_weight, self.bias, self.stride, self.padding
        )
        return out

class WSConvTranspose2d(nn.Module):
    """Weight Scale付きTranspose Convolution Layer
    """
    def __init__(
        self,
        in_channels: torch.int32,
        out_channels: torch.int32,
        kernel_size: torch.int32,
        stride: torch.int32,
        padding: torch.int32,
        gain=np.sqrt(2)
    ):
        """コンストラクタ
        WeightScale制御を付加した転置畳み込みレイヤを定義する
        転置畳み込みを使うこと以外は、WSConv2dと同様

        Args:
            in_channels (torch.int32): 入力データのチャンネル数
            out_channels (torch.int32): 出力データのチャンネル数
            kernel_size (torch.int32): カーネルサイズ
            stride (torch.int32): ストライド
            padding (torch.int32): パディング
            gain (torch.float32, optional): ハイパーパラメータ. Defaults to np.sqrt(2).
        """
        super().__init__()
        
        # Weight(filter)の初期値
        weight = torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        init.normal_(weight)
        self.weight = nn.Parameter(weight)

        # weightに乗算するscaleの値を決定(weight自身を更新するわけではない)
        scale = gain / np.sqrt(in_channels * kernel_size * kernel_size)
        scale = torch.tensor(scale, dtype=torch.float32)
        self.register_buffer('scale', scale)

        # biasの初期値
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        self.stride = stride
        self.padding = padding
    
    def forward( self, x:torch.Tensor ) -> torch.Tensor:
        """順方向伝搬
        Weight Scaleされた重みを利用して転置畳み込み演算を行う

        Args:
            x (torch.Tensor): このレイヤの入力Tensor

        Returns:
            torch.Tensor: 転置畳み込み演算後のテンソル
        """
        scaled_weight = self.weight * self.scale
        out = F.conv_transpose2d(
            x, scaled_weight, self.bias, self.stride, self.padding
        )
        return out

class AdaIN(nn.Module):
    """Adaptive Instance Normalization (AdaIN)
    Synthesis Networkで用いられる
    
    See: https://gangango.com/2019/06/16/post-573/
    See: https://qiita.com/Hiroaki-K4/items/bd4ea4e74200cbd277de
    Instance Normalizationの発展手法.
    コンテンツの特徴量の平均と分散をスタイルの特徴量の平均と分散に合わせる.
    (Instance Norm.は, 入力画像のチャネルごとの平均・分散を合わせる手法だった)
    """
    def __init__( self, dim:torch.int32, w_dim:torch.int32 ):
        """コンストラクタ
        AdaINの変換器を定義する.

        Args:
            dim (torch.int32): コンテンツ画像のチャンネル数
            w_dim (torch.int32): スタイル画像のチャンネル数
        """
        super().__init__()
        
        self.dim = dim
        self.epsilon = 1e-8
        
        # ウェイトスケーリング付きのLinearを定義するのが面倒だったのでConv1x1で代用している
        self.scale_transform = WSConv2d(
            w_dim, dim, kernel_size=1, stride=1, padding=0, gain=1
        )
        self.bias_transform = WSConv2d(
            w_dim, dim, kernel_size=1, stride=1, padding=0, gain=1
        )
    
    def forward( self, x:torch.Tensor, w:torch.Tensor ) -> torch.Tensor:
        """順方向伝搬
        AdaINを反映する.

        Args:
            x (torch.Tensor): コンテンツ画像を表すテンソル
            w (torch.Tensor): スタイル画像を表すテンソル

        Returns:
            torch.Tensor: AdaINを適用したテンソル
        """
        x = F.instance_norm(x, eps=self.epsilon)
        
        # scale
        scale = self.scale_transform(w)
        bias = self.bias_transform(w)
        
        return scale * x + bias

class NoiseLayer(nn.Module):
    """Noiseを入力に加算するためのレイヤ
    Synthesis Networkで用いられる
    """
    def __init__( self, dim:torch.int32, size:torch.int32 ):
        """コンストラクタ
        入力テンソルにノイズを加える処理を定義する
        初期状態ではnoiseの加算割合はゼロ.
        訓練により, 加算割合が更新される.

        Args:
            dim (torch.int32): 入力テンソルのチャンネル数
            size (torch.int32): 入力テンソルのサイズ(横幅/縦幅)
        """
        super().__init__()
        
        self.fixed = False
        
        self.size = size
        
        # Fixed Noise使用時の固定ノイズ設定
        noise = torch.randn([1, 1, size, size])
        self.register_buffer('fixed_noise', noise)
        
        noise_scale = torch.zeros(1, dim, 1, 1)
        self.noise_scale = nn.Parameter(noise_scale)
    
    def forward( self, x:torch.Tensor ) -> torch.Tensor:
        """順方向伝搬
        入力に対し, noise_scaleの割合でnoiseを加算する.

        Args:
            x (torch.Tensor): このレイヤの入力テンソル

        Returns:
            torch.Tensor: noiseを加算したテンソル
        """
        batch_size = x.shape[0]
        
        if self.fixed:
            noise = self.fixed_noise.expand(batch_size, -1, -1, -1)
        else:
            noise = torch.randn([batch_size, 1, self.size, self.size])
        
        noise = noise.to(device=x.device, dtype=x.dtype)
        
        out = x + noise * self.noise_scale
        return out
        
