"""
学習の中間生成物の出力を行う.
"""
import json

from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter

from model import Generator, Discriminator
from utils import create_test_z
from image_convert import RGBConverter, YUVConverter

class ArtifactMgr:
    """学習に関連する出力データを管理する
    進捗状況、現時点での画像生成の品質、メモリ使用率の推移など.
    """
    def __init__( self, settings=None ):
        """コンストラクタ
        外部にデータを出力する準備を行う.
        具体的には, 
        - ファイルの出力先ディレクトリの作成
        - tensorboardXのwriterを準備(訓練過程の可視化)
        - 画像出力時のRGB/YUVコンバータのインスタンス化
        を行う.

        Args:
            settings (dict): ハイパーパラメータを指定する辞書.
        """
        # settings
        self.settings = settings
        
        # make directory to output files
        output_dir = Path(__file__).parent.joinpath('./output')
        if not output_dir.exists():
            output_dir.mkdir()
        
        t_zone = timezone(timedelta(hours = +9), 'JST')
        self.output_root = output_dir.joinpath(
            datetime.now(t_zone).strftime("%Y%m%d_%H%M%S")
        )
        if not self.output_root.exists():
            self.output_root.mkdir()
        
        # tensorboard
        self.writer = SummaryWriter(str(self.output_root))
        
        # weights
        self.weights_root = self.output_root.joinpath('weights')
        if not self.weights_root.exists():
            self.weights_root.mkdir()
        
        # converter
        if self.settings['use_yuv']:
            self.converter = YUVConverter()
        else:
            self.converter = RGBConverter()
    
    def set_evaluate(
        self,
        device: torch.device,
        dtype: torch.dtype,
        label_size: int
    ):
        """学習器の評価に用いるパラメータを指定する.
        本メソッドで外部に出力するものはない.

        Args:
            device (torch.device): モデル評価の演算デバイスを指定する.
              評価演算では, 訓練とは別のメモリ空間を必要とするため, メモリオーバーに注意.
            dtype (torch.dtype): モデル評価で用いる値の型を指定する.
              NVIDIA apexを有効にしている場合はtorch.float16が使用できる.
              それ以外の場合は, torch.float32のみ指定可能.
            label_size (int): ラベルの分類の数を指定する.
        """
        self.device = device
        self.dtype = dtype
        
        z_dim = self.settings["network"]["z_dim"]
    
        # For evaluate
        self.test_rows = 12
        self.test_cols = 6
        test_zs = create_test_z(self.test_rows, self.test_cols, z_dim)
        self.test_z0 = torch.from_numpy(test_zs[0]).to(device, dtype)
        self.test_z1 = torch.from_numpy(test_zs[1]).to(device, dtype)
        self.test_labels0 = torch.randint(
            0, label_size, (1, self.test_cols),
            device=device
        ) # -> shape : (1, test_cols), value range : [0, label_size]
        self.test_labels0 = self.test_labels0.repeat(self.test_rows, 1)
        self.test_labels1 = torch.randint(
            0, label_size, (self.test_rows, self.test_cols),
            device=device
        ).view(-1)
    
    def dump_progress(
        self,
        level: int,
        step: int,
        g_loss: float,
        d_loss: float,
        wd=None
    ):
        """学習の進捗状況をtensorboardに出力する.

        Args:
            level (int): 解像度アップの段階(1以上の整数)
            step (int): 現在の反復回数
            g_loss (float): Generatorの損失(Tensorではなく, floatで設定すること)
            d_loss (float): Discriminatorの損失(Tensorではなく, floatで設定すること)
            wd (float, optional): Discriminatorの損失の一部を構成するパラメータ. Defaults to None.
              訓練画像によるスコアの平均から, 生成画像によるスコアの平均を引いた値.
              損失関数でwgan lossを使用している場合のみ指定.
        """
        self.writer.add_scalar(
            f"lv{level}/loss_gen", g_loss, global_step=step
        )
        self.writer.add_scalar(
            f"lv{level}/loss_disc", d_loss, global_step=step
        )
        if wd is not None:
            self.writer.add_scalar(
                f"lv{level}/wd", wd, global_step=step
            )
    
    def dump_histogram(
        self,
        step: int,
        generator: Generator,
        discriminator: Discriminator
    ):
        """パラメータのヒストグラムをtensorboardに出力する.
        ヒストグラムとは, パラメータ内の値の分布が, 
        反復回数の進み具合によってどのように変化していったかを表すグラフである.
        反復回数を増やしても値の分布があまり変化していない場合, 学習が進んでいないことを表す.

        Args:
            step (int): 学習の反復回数. ヒストグラムの軸として利用する.
            generator (Generator): 生成器. 学習器で使用しているモデルを指定する.
            discriminator (Discriminator): 識別器. 学習器で使用しているモデルを指定する.
        """
        # generator
        generator.write_histogram(
            self.writer, step
        )
        
        # discriminator
        for name, param in discriminator.named_parameters():
            self.writer.add_histogram(
                f"disc/{name}",
                param.cpu().data.numpy(),
                step
            )
    
    def dump_image(
        self,
        level: int,
        step: int,
        parameters: dict, # gsのパラメータ
        fading: bool, # fadingの実行中フラグ
        alpha: float, # fadingの混合係数,
        trues=None # 教師画像
    ):
        """現在のGeneratorでImageを生成し、その結果をtensorboardに転送する.
        Generatorはtrainerで使用しているものではなく、
        同じパラメータをコピーした別のGeneratorを構築して使用する.
        
        動作確認用に正解画像を出力することも可能.

        Args:
            level (int): 解像度アップのレベル(1以上の値). 出力のラベル付けに用いる.
            step (int): 学習の反復回数. 出力のラベル付けに用いる.
            parameters (dict): 生成器のパラメータ.
              state_dict()で取り出した辞書をそのまま設定する.
            trues (torch.Tensor, optional): 教師画像のバッチ. Defaults to None.
              学習器の動作確認(デバッグ)用に用いる.
              設定しない場合は, 教師画像はtensorboardに出力しない.
        """
        fading_text = "fading" if fading else "stabilizing"
        with torch.no_grad():
            label_size = len(self.settings["labels"])
            eval_gen = Generator(
                self.settings['network'],
                label_size
            ).to(self.device, self.dtype).eval()
            eval_gen.load_state_dict(parameters)
            eval_gen.synthesis_module.set_noise_fixed(True)
            
            # test_z0 (補間)
            fakes = eval_gen.forward(self.test_z0, self.test_labels1, alpha)
            fakes = torchvision.utils.make_grid(fakes, nrow=self.test_cols, padding=0)
            fakes = fakes.to(torch.float32).cpu().numpy()
            fakes = self.converter.from_generator_output(fakes)
            self.writer.add_image(
                f"lv{level}_{fading_text}/intpl",
                torch.from_numpy(fakes),
                step
            )
            
            # test_z1 (ランダム)
            fakes = eval_gen.forward(self.test_z1, self.test_labels1, alpha)
            fakes = torchvision.utils.make_grid(fakes, nrow=self.test_cols, padding=0)
            fakes = fakes.to(torch.float32).cpu().numpy()
            fakes = self.converter.from_generator_output(fakes)
            self.writer.add_image(
                f"lv{level}_{fading_text}/random",
                torch.from_numpy(fakes),
                step
            )
            
            del eval_gen
            
            # True image
            if trues is not None:
                trues = torchvision.utils.make_grid(trues, nrow=1, padding=0)
                trues = trues.to(torch.float32).cpu().numpy()
                trues = self.converter.from_generator_output(trues)
                self.writer.add_image(
                    f"lv{level}/true",
                    torch.from_numpy(trues),
                    step
                )
    
    
    def dump_memory_usage(
        self,
        step: int
    ):
        """現在の反復回数におけるGPUメモリ使用率をtensorboardに出力する.
        CUDAを使用している場合のみ有効.

        Args:
            step (int): 学習の反復回数.
        """
        if self.settings['use_cuda']:
            self.writer.add_scalar(
                "memory_allocated(MB)",
                torch.cuda.memory_allocated() / (1024 * 1024),
                global_step=step
            )
    
    def save_model(
        self,
        level: int,
        step: int,
        model_list: list
    ):
        """モデルのパラメータをファイルに保存する.
        保存したモデルは訓練環境再現, 学習済みのモデル利用などに用いる.

        Args:
            level (int): 解像度アップのレベル. 保存されるモデルの格納ディレクトリ名に用いる.
            step (int): 現在の反復回数. 保存されるモデルの格納ディレクトリ名に用いる.
            model_list (list): 保存するモデルのパラメータと, 保存ファイル名のタプルをリストにしたもの.
              形式は次のようになる.
              [
                  ([モデル1のパラメータ], [モデル1のパラメータを保存する際のファイル名]),
                  ([モデル2のパラメータ], [モデル2のパラメータを保存する際のファイル名]),
                  ...
                  ([モデルnのパラメータ], [モデルnのパラメータを保存する際のファイル名]),
              ]
              モデルのパラメータは, state_dict()メソッドにより取得する.
        """
        savedir = self.weights_root.joinpath(f"{step}_lv{level}")
        if not savedir.exists():
            savedir.mkdir()
        for param, fname in model_list:
            torch.save(param, savedir.joinpath(fname))
    
    def save_settings( self ):
        """現在の設定ファイルの内容をコピーしてファイルに出力する.
          訓練環境の再現に用いることを想定.

        Raises:
            ValueError: settingsが未指定の場合
        """
        if self.settings is None:
            raise ValueError('settings is None')
        
        json_path = self.output_root.joinpath('settings.json')
        path_str = str(json_path)
        
        with open(path_str, 'w') as f:
            json.dump( self.settings, f, indent=4 )
    