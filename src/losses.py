import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
from absl import flags

FLAGS = flags.FLAGS


def custom_loss_func(y: Tensor, y_pred: Tensor) -> Tensor:
    """カスタムの損失関数を実装する.

    Args:
        y {Tensor} -- 例えば教師ラベル
        y_pred {Tensor} -- 例えばモデルの予測値

    Returns:
        Tensor -- 損失の計算結果
    """
    loss = y - y_pred
    return loss
