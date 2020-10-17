"""
numpy配列で表現された画像の変換を行う機能を提供

RGB画像を入力とし, それをdatasetとして扱える形にする.
または, Generatorからの出力を画像として表示できる形にする.
"""
import numpy as np

class RGBConverter:
    """RGB画像の変換器
    
    各要素が[0, 1]である画像を[-1, 1]の範囲にスケーリングする(データセット化).
    または,
    各要素が[-1, 1]である画像を[0, 1]の範囲にスケーリングする(画像化).
    """
    @staticmethod
    def to_train_data( images: np.ndarray ) -> np.ndarray:
        """画像をデータセットとして使用できる形式に変換する.

        Args:
            images (np.ndarray): 入力画像(RGB形式).
              各要素の値の範囲が[0, 1]であるテンソル.

        Returns:
            np.ndarray: 変換後の画像(RGB形式).
              各要素の値の範囲が[-1, 1]であるテンソル.
        """
        # images is in [0, 1]
        return images * 2 - 1
    
    @staticmethod
    def from_generator_output( images: np.ndarray ) -> np.ndarray:
        """Generatorが生成した画像を描画できる形式に変換する.

        Args:
            images (np.ndarray): 入力画像(RGB形式).
              各要素の値の範囲が[-1, 1]であるテンソル

        Returns:
            np.ndarray: 変換後の画像(RGB形式).
              各要素の値の範囲が[0, 1]であるテンソル.
        """
        # images is in (about) [-1, 1]
        out = (images + 1) / 2
        # 範囲が[0, 1]でない値を最小値0, 最大値1になるようにクリップする.
        out = np.clip( out, 0, 1 )
        return out

class YUVConverter:
    """YUV画像の変換器
    
    RGBの元画像をYUVに変換した上で[-1, 1]の範囲にスケーリングする.
    または,
    [-1, 1]の範囲のYUV画像をclip, スケーリングした上で, RGBに変換する.
    
    YUVとRGBの変換式は
    https://qiita.com/chilchil0/items/4e553bead1270054de1d
    を参照
    """
    # ITU-R BT.601
    # YCbCr
    @staticmethod
    def to_train_data( images: np.ndarray ) -> np.ndarray:
        """画像をデータセットとして使用できる形式に変換する.

        Args:
            images (np.ndarray): 入力画像(RGB形式).
              各要素の値の範囲が[0, 1]であるテンソル.

        Returns:
            np.ndarray: 変換後の画像(YUV形式).
              各要素の値の範囲が[-1, 1]であるテンソル.
        """
        # images is in [0, 1]
        # 出力先の領域を生成
        yuv = np.zeros_like(images, dtype=np.float)
        
        # Y
        yuv[:, 0] = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        # Cb
        yuv[:, 1] = -0.168736 * images[:, 0] - 0.331264 * images[:, 1] + 0.5 * images[:, 2]
        # Cr
        yuv[:, 2] = 0.5 * images[:, 0] - 0.418688 * images[:, 1] - 0.081312 * images[:, 2]
        
        # 値の範囲を[-1, 1]にScalingする
        yuv[:, 0] = yuv[:, 0]*2 - 1
        yuv[:, 1:] *= 2
        
        return yuv
    
    @staticmethod
    def from_generator_output( images: np.ndarray ) -> np.ndarray:
        """Generatorが生成した画像を描画できる形式に変換する.

        Args:
            images (np.ndarray): 入力画像(YUV形式).
              各要素の値の範囲が[-1, 1]であるテンソル

        Returns:
            np.ndarray: 変換後の画像(RGB形式).
              各要素の値の範囲が[0, 1]であるテンソル.
        """
        # images is in [-1, 1]
        
        # [-1, 1]の範囲外にある値を最小値-1, 最大値1となるようにclipする.
        images = images.copy()
        images = np.clip(images, -1, 1)
        
        # 値の範囲を[0, 1]にScalingする
        images[:, 0] = (images[:, 0] + 1)/2
        images[:, 1:] /= 2

        # 出力先の領域を生成
        rgb = np.zeros_like(images, dtype=np.float)
        
        # R
        rgb[:, 0] = images[:, 0] + 1.402 * images[:, 2]
        # G
        rgb[:, 1] = images[:, 0] - 0.344136 * images[:, 1] - 0.714136 * images[:, 2]
        # B
        rgb[:, 2] = images[:, 0] + 1.772 * images[:, 1]
        
        # 再度clip
        rgb = np.clip(rgb, 0, 1)
        
        return rgb