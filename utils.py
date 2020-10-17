"""
評価用の入力ノイズベクトルを生成する機能を提供する.
"""
import numpy as np


def slerp(
    val: float,
    low: np.ndarray,
    high: np.ndarray
) -> np.ndarray:
    """球面線形補間
    
    lowとhighを球面線形補間で混合したベクトルを生成する.
    混合割合をvalで指定する.

    Args:
        val (float): 混合割合. 
          1に近づくほどhighの割合が大きくなる. 0に近づくほどlowの割合が大きくなる.
        low (np.ndarray): 混合するベクトルA
        high (np.ndarray): 混合するベクトルB

    Returns:
        np.ndarray: 球面線形補間により混合されたベクトル
    """
    omega = np.arccos(
        np.dot(
            low / np.linalg.norm(low),
            high / np.linalg.norm(high)
        )
    )
    so = np.sin(omega)
    out = np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high
    return out


def create_z( size: int, dim: int ) -> np.ndarray:
    """ノイズベクトルを生成する.
    
    生成されるノイズは, 標準ガウス分布に従う.
    batch_sizeをsize, ベクトルの長さをdimで指定する.

    Args:
        size (int): ノイズベクトルのバッチサイズ
        dim (int): ノイズベクトルの次元(長さ)

    Returns:
        np.ndarray: 生成したノイズベクトル
    """
    z = np.random.normal( 0, 1, [size, dim] )
    return z


def create_test_z( rows: int, cols: int, dim: int ) -> tuple:
    """GANの補間動作を評価するための入力ベクトルを生成する
    
    まず, ランダムに生成したノイズベクトルz1_startとz1_endを生成する.
    補完を確認するため, z1_startからz1_endまでの間に存在するベクトルを
    球面線形補間によって取得し, それらをz1としてまとめる.
    
    また, 完全にランダムに作成したノイズベクトルのセットであるz2も生成する.
    
    これらz1とz2のセットをtupleとしてReturnする.

    Args:
        rows (int): 行数.
          この値の数だけ, 補間の中間となるサンプルを生成する.
          (補間サンプルを生成するのはz1のみである.
           z2に対しては, 別々のサンプルを生成する.)
        cols (int): 列数.
          この値の数だけ, 全く別のサンプルを作る.
        dim (int): ノイズベクトルの次元数(長さ), z_dim.

    Returns:
        tuple: 生成された入力ベクトルをまとめたtuple.
          1要素目が補間を含む入力ベクトル.
          2要素目が全要素ランダムに生成されたベクトル.
          どちらも型はnp.ndarray
    """
    # interpolation
    z1 = np.zeros([rows, cols, dim])
    z1_start = create_z(cols, dim)
    z1_end = create_z(cols, dim)
    for i in range(rows):
        val = i / (rows-1)
        for j in range(cols):
            z1[i, j] = slerp(val, z1_start[j], z1_end[j])
    z1 = z1.reshape([-1, dim])

    # random
    z2 = create_z(rows * cols, dim)

    return z1, z2