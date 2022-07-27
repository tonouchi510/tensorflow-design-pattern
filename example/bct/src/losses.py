import tensorflow as tf
import functools
from absl import flags

FLAGS = flags.Flag


def bct_loss_func(y_true, y_pred, y_true_old, y_pred_old, alpha=1.0):
    """BCT 損失関数.

    Args:
        y_true (_type_): 教師ラベル
        y_pred (_type_): 分類器の出力
        y_true_old (_type_): 旧分類器の教師ラベル
        y_pred_old (_type_): 旧分類器の出力
        alpha (float, optional): 旧分類器の誤差の重み. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    l_new = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    l_old = tf.keras.losses.categorical_crossentropy(y_true_old, y_pred_old)
    loss = l_new + alpha * l_old
    return loss


loss_func = functools.partial(
    bct_loss_func,
    alpha=FLAGS.alpha
)
