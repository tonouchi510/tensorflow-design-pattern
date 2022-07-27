import tensorflow as tf
import functools
from absl import flags

FLAGS = flags.Flag


def custom_loss_func(y_true, y_pred, delta=1.0):
    """カスタムの損失関数の実装

    ここでは、適当にCCEEにHuberLossっぽい要素を追加してみた例

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        delta (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: lossの値
    """
    error = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    abs_error = tf.abs(error)
    loss = tf.where(condition=abs_error <= delta, x=0.5 * tf.square(error), y=delta * abs_error - 0.5 * tf.square(delta))
    return loss


loss_func = functools.partial(
    custom_loss_func,
    delta=FLAGS.delta
)
