import tensorflow as tf
from typing import Callable


# 損失関数用（以下例：適当にHuberLossっぽい要素を追加してみた例）
def get_loss_func(delta=1.0) -> Callable:
    """指定したパラメータで、損失関数を返す

    Args:
        delta (float, optional): _description_. Defaults to 1.0.

    Returns:
        Callable: カスタムの損失関数
    """
    def custom_loss_func(y_true, y_pred):
        error = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        abs_error = tf.abs(error)
        loss = tf.where(condition=abs_error <= delta, x=0.5 * tf.square(error), y=delta * abs_error - 0.5 * tf.square(delta))
        return loss

    return custom_loss_func
