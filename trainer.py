"""
Style-GANの訓練を行う
"""
from apex import amp
import torch
import torch.optim as optim
import torchvision

# My class
from artifact import ArtifactMgr
from data_loader import LabeledDataLoader
from utils import create_z

class Trainer:
    """学習器クラス
    学習に必要な各種オブジェクトは外で生成する.
    また, 学習時に生成される中間生成物はArtifactMgrを通じてtensorboardXに出力される.
    """
    def __init__(
        self,
        settings: dict,
        artifacts: ArtifactMgr,
        devices: dict,
        models: dict,
        losses: dict,
        optimizers: dict,
        loader: LabeledDataLoader,
        converter: any
    ):
        """コンストラクタ
        学習に必要なパラメータを受け取り準備を行う.

        Args:
            settings (dict): ハイパーパラメータを指定する辞書.
            artifacts (ArtifactMgr): 学習器が出力する中間生成物を管理するモジュール.
            devices (dict): 学習環境(演算を行うデバイス, 演算の型)を指定する辞書.
              下記のキーを含むこと.
              - amp_handle (apex.amp.handle.AmpHandle or apex.amp.handle.NoOpHandle): NVIDIA apexのハンドル.
              - device (torch.device): 学習時の演算デバイス.
                'cpu'または'cuda:0'のデバイスを指定したtorch.deviceオブジェクトを指定.
              - dtype (torch.dtype): 学習時の演算の型.
                NVIDIA apexを使用する場合は, torch.float16が使用可能.
                それ以外の場合は, torch.float32を指定する.
              - test_device (torch.device): 評価時の演算デバイス.
                'cpu'または'cuda:0'のデバイスを指定したtorch.deviceオブジェクトを指定.
                評価に用いるモデルは別のメモリ空間を必要とするため,
                メモリが足りない場合は, test_deviceのみcpuにするなどの対処をすること.
              - test_dtype (torch.dtype): 評価時の演算の型. troch.float32を指定する.
            models (dict): Style GANのモデルを格納した辞書.
              下記のキーを含むこと
              - generator (Generator): Style GANの生成器モデル
              - discriminator (Discriminator): Style GANの識別器モデル
              - gs (Generator): Gの移動平均を算出するための生成器モデル.
                generatorに指定したモデルと同モデルで別のインスタンスのものを指定する.
            losses (dict): 損失関数を格納した辞書.
              下記のキーを含むこと
              - d_lossf (function): 識別器の損失関数(関数オブジェクト)
              - g_lossf (function): 生成器の損失関数(関数オブジェクト)
            optimizers (dict): 最適化器を格納した辞書.
              下記のキーを含むこと
              - d_opt (torch.optim.Optimizer): 識別器の最適化器
              - g_opt (torch.optim.Optimizer): 生成器の最適化器
            loader (LabeledDataLoader): データセットからバッチ単位で訓練データを取得できるDataLoaderオブジェクト.
            converter (RGBConverter or YUVConverter): 画像を訓練データに変換, または, 生成データを画像に変換する変換器.
        """
        # Hyper parameters
        self.settings = settings
        # Artifacts
        self.artifact_mgr = artifacts
        # Devices
        self.amp_handle = devices['amp_handle']
        self.device = devices['device']
        self.dtype = devices['dtype']
        self.test_device = devices['test_device']
        self.test_dtype = devices['test_dtype']
        # Model
        self.generator = models['generator']
        self.discriminator = models['discriminator']
        self.gs = models['gs']
        self.gs_beta = self.settings['gs_beta']
        # Loss functions
        self.d_lossf = losses['d_lossf']
        self.g_lossf = losses['g_lossf']
        # Optimizers
        self.d_opt = optimizers['d_opt']
        self.g_opt = optimizers['g_opt']
        # Datasets
        self.loader = loader
        # Image converter
        self.converter = converter
    
    def train( self ):
        """学習を実行する.
        訓練過程は, ArtifactMgrを通してtensorboardXに出力する.
        """
        # Parameters
        #fading = False
        fading = False
        alpha = 1
        step = 0
        
        z_dim = self.settings["network"]["z_dim"]
        level = self.settings['start_level']
        batch_sizes = self.settings['batch_sizes']
        num_images_in_stage = self.settings['num_images_in_stage']
        
        # For evaluate
        self.artifact_mgr.set_evaluate(self.test_device, self.test_dtype, self.loader.label_size)
        
        for loop in range(self.settings['max_loop']):
            size = 2 ** (level + 1)
            
            batch_size = batch_sizes[level - 1]
            alpha_delta = batch_size / num_images_in_stage
            
            image_count = 0
            
            for batch, labels in self.loader.generate(batch_size, size, size):
                # Pre train
                step += 1
                image_count += batch_size
                if fading:
                    alpha = min(1.0, alpha + alpha_delta)
                
                # data
                batch = batch.transpose([0, 3, 1, 2]) # --> [B, C, W, H]に並び替え
                batch = self.converter.to_train_data(batch)
                trues = torch.from_numpy(batch).to(self.device, self.dtype)
                labels = torch.from_numpy(labels).to(self.device)
                
                # rest
                self.g_opt.zero_grad()
                self.d_opt.zero_grad()
                
                # === train discriminator ===
                z = create_z(batch_size, z_dim)
                z = torch.from_numpy(z).to(self.device, self.dtype)
                fakes = self.generator.forward(z, labels, alpha)
                fakes_nograd = fakes.detach()
                
                for param in self.discriminator.parameters():
                    param.requires_grad_(True)
                
                loss_ret = self.d_lossf(
                    self.discriminator, trues, fakes_nograd, labels, alpha
                )
                d_loss = loss_ret[0]
                if (len(loss_ret) >= 2):
                    wd = loss_ret[1]
                else:
                    wd = None
                
                #d_loss.backward()
                with self.amp_handle.scale_loss(d_loss, self.d_opt) as scaled_loss:
                    scaled_loss.backward()
                self.d_opt.step()
                
                # === train generator ===
                z = create_z(batch_size, z_dim)
                z = torch.from_numpy(z).to(self.device, self.dtype)
                fakes = self.generator.forward(z, labels, alpha)
                
                for param in self.discriminator.parameters():
                    param.requires_grad_(False)
                
                loss_ret = self.g_lossf(self.discriminator, fakes, labels, alpha)
                g_loss = loss_ret[0]
                
                # g_loss.backward()
                with self.amp_handle.scale_loss(g_loss, self.g_opt) as scaled_loss:
                    scaled_loss.backward()
                    del scaled_loss
                
                self.g_opt.step()
                
                #del trues, fakes, fakes_nograd
                del fakes, fakes_nograd
                
                # update gs
                for gparam, gsparam in zip(self.generator.parameters(), self.gs.parameters()):
                    gsparam.data = (1 - self.gs_beta) * gsparam.data + \
                                   self.gs_beta * gparam.data
                self.gs.w_average.data = (1 - self.gs_beta) * self.gs.w_average.data +\
                                         self.gs_beta * self.generator.w_average.data
                
                # log
                if step % 1 == 0:
                    print(f"[lv:{level:2d}][step:{step:7d}][imgs:{image_count:7d}] "
                          f"alpha: {alpha:.5f} "
                          f"G-loss: {g_loss.item():+11.7f} "
                          f"D-loss: {d_loss.item():+11.7f} ")
                    
                    self.artifact_mgr.dump_progress(
                        level, step, g_loss.item(), d_loss.item(), wd
                    )
                
                del d_loss, g_loss
                
                # histogram
                hist_step = self.settings['save_steps']['histogram']
                if hist_step > 0 and step % hist_step == 0:
                    self.artifact_mgr.dump_histogram(
                        step, self.gs, self.discriminator
                    )
                
                # image
                image_step = self.settings['save_steps']['image']
                if (image_step > 0 and step % image_step == 0) or alpha == 0:
                    self.artifact_mgr.dump_image(
                        level, step, self.gs.state_dict(), fading, alpha, trues
                    )
                    
                    # memory usage
                    self.artifact_mgr.dump_memory_usage(step)
                
                del trues
                
                # model save
                model_step = self.settings['save_steps']['model']
                if step % model_step == 0 and level >= 5 and not fading:
                    self.artifact_mgr.save_model(level, step, [
                        (self.generator.state_dict(), 'gen.pth'),
                        (self.gs.state_dict(), 'gs.pth'),
                        (self.discriminator.state_dict(), 'disc.pth')
                    ])
                
                # switch fading/stabilizing
                if image_count > num_images_in_stage:
                    if fading:
                        print('start stabilizing')
                        fading = False
                        alpha = 1
                        image_count = 0
                    elif level < self.settings['max_level']:
                        print(f"end lv: {level}")
                        break
                
                if self.settings['debug_mode']:
                    break
            
            # level up
            if level < self.settings['max_level']:
                level += 1
                self.generator.set_level(level)
                self.discriminator.set_level(level)
                self.gs.set_level(level)
                
                fading = True
                alpha = 0
                print(f"lv up: {level}")
                
                if self.settings['reset_optimizer']:
                    self.g_opt = optim.Adam(
                        [
                            {
                                "params": self.generator.latent_transform.parameters(),
                                "lr": self.settings["learning_rates"]["latent_transformation"]
                            },
                            {
                                "params": self.generator.synthesis_module.parameters()
                            }
                        ],
                        lr=self.settings["learning_rates"]["generator"],
                        betas=(0.0, 0.99),
                        eps=1e-8
                    )
                    self.d_opt = optim.Adam(
                        self.discriminator.parameters(),
                        lr=self.settings["learning_rates"]["discriminator"],
                        betas=(0.0, 0.99),
                        eps=1e-8
                    )
