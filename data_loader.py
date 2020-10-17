"""
Datasetをファイルからロードし、データセットとして扱えるようにする機能を提供
"""
import random
from pathlib import Path
import numpy as np
from PIL import Image
import concurrent.futures as futures

def chunks( target_list: list, chunk_size: int ):
    """リストの要素をchunk_size要素ごとに順番に切り出す

    Args:
        target_list (list): 切り出す対象のlist
        chunk_size (int): 切り出す要素の間隔

    Yields:
        list: 切り出されたlistの一部(これもlistである)
    """
    num_chunks = len(target_list) // chunk_size
    for idx_chunk in range(num_chunks):
        start = idx_chunk * chunk_size
        yield target_list[start : start + chunk_size]

def crop_center(
    pil_img: Image,
    crop_width: int,
    crop_height: int
) -> Image:
    """画像の中心から指定したサイズで切り抜く
    See: https://note.nkmk.me/python-pillow-image-crop-trimming/

    Args:
        pil_img (Image): 切り抜く対象の画像(PIL Image)
        crop_width (int): 切り抜き幅
        crop_height (int): 切り抜き高さ

    Returns:
        Image: 指定したサイズで切り抜いた画像
    """
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                        (img_height - crop_height) // 2,
                        (img_width + crop_width) // 2,
                        (img_height + crop_height) // 2))

class TrainDataLoader:
    """データセットをロードする.
    """
    def __init__( self, paths: list, settings=None ):
        """コンストラクタ
        パスのリストからデータセットを構築する

        Args:
            paths (list): 画像ファイルのパスのリスト.
            settings (dict, optional): ハイパーパラメータを指定する辞書. Defaults to None.
        """
        self.paths = paths
        
        # Data settings
        self.flip = True
        self.color_shift = False
        if settings is not None:
            self.flip = settings['flip']
            self.color_shift = settings['color_shift']
    
    def generate( self, batch_size: int, width: int, height: int ):
        """batch_sizeで指定した個数の画像データをロードするgeneratorを生成する.

        Args:
            batch_size (int): バッチサイズ.
              この値の数だけ一度に画像が出力されるgeneratorを生成する.
            width (int): 読み込んだ後の画像の横幅をこの値にする.
              元画像の横幅ではない.
            height (int): 読み込んだ後の画像の縦幅をこの値にする.
              元画像の縦幅ではない.

        Yields:
            np.ndarray: generatorから出力される画像のバッチ
        """
        num_workers = 5
        
        with futures.ThreadPoolExecutor(num_workers) as executer:
            tasks = []
            while True:
                np.random.shuffle(self.paths)
                for paths in chunks(self.paths, batch_size):
                    newtask = executer.submit(self.load_images, paths, width, height)
                    tasks.append(newtask)
                    if len(tasks) < num_workers:
                        continue
                    task = tasks.pop(0)
                    result = task.result(100)
                    if result is not None:
                        yield result
    
    def load_images( self, paths: list, width: int, height: int ) -> np.ndarray:
        """パスのリストに含まれる画像を全てロードし, numpy配列に格納する.
        
        画像はwidth, heightで指定したサイズにリサイズする.
        画像が指定したサイズではない場合, 画像の中心から指定したサイズになるようクロップされる.
        画像の値の範囲は, [0, 1]になるようにスケーリングされる.
        (元画像に戻す場合は255倍にし, 整数値にキャストする)
        
        Args:
            paths (list): 読み込む画像の配置されているパスのリスト
            width (int): 読み込んだ後の画像の横幅をこの値にする.
              元画像の横幅ではない.
            height (int): 読み込んだ後の画像の縦幅をこの値にする.
              元画像の縦幅ではない.

        Returns:
            np.ndarray: 読み込んだ画像が格納されているnumpy配列
              配列の形状は, [len(paths), height, width, channel_num]となる.
        """
        array = np.empty([len(paths), height, width, 3])
        
        for idx_path, path in enumerate(paths):
            try:
                image = Image.open(str(path))
            except:
                continue # Ignore path
            
            # 中央を抜き出してリサイズ
            image = crop_center(image, min(image.size), min(image.size))
            image = image.resize((width, height), Image.LANCZOS)
            
            # arrayに変換
            image = np.array(image, dtype=np.float32)
            image /= 255
            
            # 左右反転
            if self.flip:
                if bool(random.getrandbits(1)):
                    image = np.flip(image, 1)
            
            # color_shift
            if self.color_shift:
                v = np.random.choice([0.7, 1.0, 1.3])
                image = image ** v
            
            array[idx_path] = image
        
        return array

class LabeledDataLoader:
    def __init__( self, settings=None ):
        """コンストラクタ
        特定のディレクトリからラベル付きデータセットを構築する

        Args:
            settings (dict, optional): ハイパーパラメータを指定する辞書. Defaults to None.
        """
        # Data settings
        self.flip = True
        self.color_shift = False
        self.labels = ['img_align_celeba'] # データセットのラベルのリストを指定
        self.joinpath = '../data/celeba' # データセットのルートを指定. ラベルの1つ上のディレクトリ
        self.target_type = '*.*' # 取得する画像のファイルタイプを指定
        if settings is not None:
            self.flip = settings['data_augmentation']['flip']
            self.color_shift = settings['data_augmentation']['color_shift']
            self.labels = settings['labels']
            self.joinpath = settings['joinpath']
            self.target_type = settings['target_type']
        
        self.label_size = len(self.labels)
        
        img_root = Path(__file__).parent.joinpath(self.joinpath)
        self.data = []
        
        for idx_chara, chara in enumerate(self.labels):
            chara_root = img_root.joinpath(chara)
            for path in chara_root.glob(self.target_type):
                self.data.append((path, idx_chara))
        
        print(len(self.data))
    
    def generate( self, batch_size: int, width: int, height: int ):
        """batch_sizeで指定した個数の画像データをロードするgeneratorを生成する.

        Args:
            batch_size (int): バッチサイズ.
              この値の数だけ一度に画像が出力されるgeneratorを生成する.
            width (int): 読み込んだ後の画像の横幅をこの値にする.
              元画像の横幅ではない.
            height (int): 読み込んだ後の画像の縦幅をこの値にする.
              元画像の縦幅ではない.

        Yields:
            np.ndarray: generatorから出力される画像のバッチ
        """
        num_workers = 5
        
        with futures.ThreadPoolExecutor(num_workers) as executer:
            tasks = []
            while True:
                np.random.shuffle(self.data)
                for data in chunks(self.data, batch_size):
                    newtask = executer.submit(self.load_images, data, width, height)
                    tasks.append(newtask)
                    if len(tasks) < num_workers:
                        continue
                    task = tasks.pop(0)
                    result = task.result(100)
                    if result is not None:
                        yield result
    
    def load_images( self, data: list, width: int, height: int ) -> np.ndarray:
        """パスのリストに含まれる画像を全てロードし, numpy配列に格納する.
        
        画像はwidth, heightで指定したサイズにリサイズする.
        画像が指定したサイズではない場合, 画像の中心から指定したサイズになるようクロップされる.
        画像の値の範囲は, [0, 1]になるようにスケーリングされる.
        (元画像に戻す場合は255倍にし, 整数値にキャストする)
        
        Args:
            data (list): 読み込む画像の配置されているパスとラベルのリスト
            width (int): 読み込んだ後の画像の横幅をこの値にする.
              元画像の横幅ではない.
            height (int): 読み込んだ後の画像の縦幅をこの値にする.
              元画像の縦幅ではない.

        Returns:
            np.ndarray: 読み込んだ画像が格納されているnumpy配列
              配列の形状は, [len(paths), height, width, channel_num]となる.
        """
        images = np.empty([len(data), height, width, 3])
        labels = np.array(list(map(lambda  x: x[1], data)))
        
        for idx_row, row in enumerate(data):
            path, label = row
            try:
                image = Image.open(str(path))
            except:
                continue # Ignore path
            
            # 中央を抜き出してリサイズ
            image = crop_center(image, min(image.size), min(image.size))
            image = image.resize((width, height), Image.LANCZOS)
            
            # arrayに変換
            image = np.array(image, dtype=np.float32)
            image /= 255
            
            # 左右反転
            if self.flip:
                if bool(random.getrandbits(1)):
                    image = np.flip(image, 1)
            
            # color_shift
            if self.color_shift:
                v = np.random.choice([0.7, 1.0, 1.3])
                image = image ** v
            
            images[idx_row] = image
        
        return images, labels
