"""
Style-GAN実行のためのシーケンスを実現
"""

from apex import amp
import torch
import torch.optim as optim

# My class
from hyper_params import HyperParams
from artifact import ArtifactMgr
from model import Generator, Discriminator
from loss import (
    d_lsgan_loss, g_lsgan_loss,
    d_wgan_loss, g_wgan_loss,
    d_logistic_loss, g_logistic_loss
)
from data_loader import LabeledDataLoader
from image_convert import RGBConverter, YUVConverter
from trainer import Trainer

class Sequence:
    """学習に必要なシーケンスを管理する.
    """
    def __init__( self ):
        """学習に必要な各種インスタンスを生成し, 学習器を動作可能な状態に構築する.
        """
        # Hyper parameters
        self.settings = HyperParams().get()
        # Prepare output directry
        artifacts = ArtifactMgr( self.settings )
        artifacts.save_settings()
        # Device settings
        amp_handle, device, dtype, test_device, test_dtype = self.prepare_device()
        self.devices = {
            'amp_handle' : amp_handle,
            'device' : device,
            'dtype' : dtype,
            'test_device' : test_device,
            'test_dtype' : test_dtype,
        }
        # Model
        generator, discriminator, gs = self.prepare_model()
        self.models = {
            'generator' : generator,
            'discriminator' : discriminator,
            'gs' : gs
        }
        
        # Loss function
        d_lossf, g_lossf = self.prepare_loss()
        self.loss = {
            'd_lossf' : d_lossf,
            'g_lossf' : g_lossf,
        }
        # Optimizer
        g_opt, d_opt = self.prepare_optimizer()
        self.optimizer = {
            'g_opt' : g_opt,
            'd_opt' : d_opt,
        }
        # Dataset (DataLoader)
        loader, converter = self.prepare_dataset()
        # Trainer
        self.trainer = Trainer(
            settings=self.settings,
            artifacts=artifacts,
            devices=self.devices,
            models=self.models,
            losses=self.loss,
            optimizers=self.optimizer,
            loader=loader,
            converter=converter
        )
    
    def run( self ):
        """学習を実行する.
        """
        self.trainer.train()
    
    def prepare_device( self ) -> tuple:
        """学習環境に関するパラメータを準備する.
        具体的には, 半精度演算の利用有無, GPU演算の利用有無を決定する.

        Returns:
            tuple: 下記の要素を順番に含むtuple
            - amp_handle (apex.amp.handle.AmpHandle or apex.amp.handle.NoOpHandle): NVIDIA apexのハンドル.
              See: https://github.com/NVIDIA/apex
              設定で'use_apex'をtrueと設定している場合, このハンドルを用いて半精度浮動小数点数演算を行うことができる.
              この場合, 'use_cuda'を同時にtrueにする必要があり, かつGPU演算版のapexがインストールされている必要がある.
              設定で'use_apex'をfalseと設定している場合は何も操作を行わないハンドルを取得する.
            - device (torch.device): 訓練用の演算を行うデバイス.
              設定で'use_cuda'をtrueに設定している場合, GPU(CUDA)で演算を行う(※推奨).
              この場合, NVIDIA製GPU, NVIDIA CUDA Toolkitと, GPU対応版PyTorchが必要.
              設定で'use_cude'をfalseに設定している場合, CPUで演算を行う.
              (CPU演算の場合, フルサイズのモデルで演算するのは困難なので, モデルやハイパーパラメータの調整を推奨)
            - dtype (torch.dtype): 訓練用の演算の型.
              現状, torch.float32で固定.
            - test_device (torch.device): 評価用の演算を行うデバイス.
              設定に関わらず, CPU演算で固定.
            - test_dtype (torch.dtype): 評価用の演算の型.
              設定に関わらず, torch.float32で固定.
        """
        settings = self.settings
        
        # NVIDIA APEX
        amp_handle = amp.init( settings['use_apex'] )
        
        # CUDA
        if settings['use_cuda']:
            device = torch.device('cuda:0')
            test_device = torch.device('cpu')
        else:
            device = torch.device('cpu')
            test_device = torch.device('cpu')
        
        # dtype
        dtype = torch.float32
        test_dtype = torch.float32
        
        return amp_handle, device, dtype, test_device, test_dtype
    
    def prepare_model( self ):
        """StyleGANのモデルを構築する.

        Returns:
            tuple: 下記の要素を順番に含むtuple
            - generator (Generator): 生成器.
            - discriminator (Discriminator): 識別器.
            - gs (Generator): Gの移動平均を実現するための第二のGenerator.
        """
        settings = self.settings
        
        label_size = len(settings["labels"])
        generator = Generator( settings["network"], label_size )
        generator = generator.to( self.devices['device'], self.devices['dtype'] )
        discriminator = Discriminator( settings["network"], label_size )
        discriminator = discriminator.to( self.devices['device'], self.devices['dtype'] )

        # long-term average
        gs = Generator( settings["network"], label_size )
        gs = gs.to( self.devices['device'], self.devices['dtype'] )
        gs.load_state_dict(generator.state_dict())
        
        # start level
        level = settings['start_level']
        generator.set_level(level)
        discriminator.set_level(level)
        gs.set_level(level)
        
        return generator, discriminator, gs
    
    def prepare_loss( self ) -> tuple:
        """生成器, 識別器の損失関数を準備する.
        損失関数は'loss'の設定により下記の3つの中から1つを選択できる.
        - 'wgan' : Wasserstein距離に基づく損失を用いる.
        - 'lsgan' : 平均二乗誤差(MSELoss)に基づく損失を用いる.
        - 'logistic' : ロジスティック損失を用いる.

        Raises:
            ValueError: 上記以外の値が設定されていた場合.

        Returns:
            tuple: 下記の要素を順番に含むtuple
            - d_lossf (function): 識別器の損失関数オブジェクト.
            - g_lossf (function): 生成器の損失関数オブジェクト.
        """
        settings = self.settings
        
        if settings['loss'] == 'wgan':
            d_lossf = d_wgan_loss
            g_lossf = g_wgan_loss
        elif settings['loss'] == 'lsgan':
            d_lossf = d_lsgan_loss
            g_lossf = g_lsgan_loss
        elif settings['loss'] == 'logistic':
            d_lossf = d_logistic_loss
            g_lossf = g_logistic_loss
        else:
            raise ValueError(f"Invalid loss: {settings['loss']}")
        
        return d_lossf, g_lossf
    
    def prepare_optimizer( self ) -> tuple:
        """訓練におけるパラメータの最適化器を準備する.
        本学習モデルではAdamを用いることとし, 動作に必要なパラメータを設定したものをReturnする.

        Returns:
            tuple: 下記の要素を順番に含むtuple
            - g_opt (torch.optim.Adam): 生成器のパラメータ更新に用いる最適化器.
            - d_opt (torch.optim.Adam): 識別器のパラメータ更新に用いる最適化器.
        """
        settings = self.settings
        
        # Learning rate
        # generatorはMapping Networkとそれ以外で学習率を変える
        lt_learning_rate = settings["learning_rates"]["latent_transformation"]
        g_learning_rate = settings["learning_rates"]["generator"]
        d_learning_rate = settings["learning_rates"]["discriminator"]
        
        # Optimizer
        g_opt = optim.Adam(
            [
                {
                    "params": self.models['generator'].latent_transform.parameters(),
                    "lr": lt_learning_rate
                },
                {
                    "params": self.models['generator'].synthesis_module.parameters()
                }
            ],
            lr=g_learning_rate,
            betas=(0.0, 0.99),
            eps=1e-8
        )
        d_opt = optim.Adam(
            self.models['discriminator'].parameters(),
            lr=d_learning_rate,
            betas=(0.0, 0.99),
            eps=1e-8
        )
        
        return g_opt, d_opt
    
    def prepare_dataset( self ):
        """データセットからバッチサイズ単位でデータを取り出せるDataLoaderと, 
        データセットの画像を学習器に投入できる状態に変換する変換器を取得する.
        (変換器は, 生成器が生成したデータを画像に変換する機能も持っている)
        
        変換器はYUVConverterとRGBConverterの2種類が存在し,
        どちらを使用するかは'use_yuv'の設定で切り替える.
        'use_yuv'がtrueの場合, YUVConverterを使用する.
        'use_yuv'がfalseの場合, RGBConverterを使用する.

        Returns:
            tuple: 下記の要素を順番に含むtuple
            - loader (LabeledDataLoader): DataLoader
            - converter (YUVConverter or RGBConverter): 画像変換器
        """
        settings = self.settings
        
        loader = LabeledDataLoader( settings )
        
        if settings['use_yuv']:
            converter = YUVConverter()
        else:
            converter = RGBConverter()
        
        return loader, converter
