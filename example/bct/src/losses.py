import tensorflow as tf


def get_loss_func(bct_alpha=1.0):
    """BCTの損失関数を返す.

    Args:
        bct_alpha (float, optional): 旧分類器の誤差の重み. Defaults to 1.0.
    """
    def bct_loss(y_true, y_pred, y_true_old, y_pred_old):
        """BCT 損失関数.

        Args:
            y_true (_type_): 教師ラベル
            y_pred (_type_): 分類器の出力
            y_true_old (_type_): 旧分類器の教師ラベル
            y_pred_old (_type_): 旧分類器の出力

        Returns:
            _type_: _description_
        """
        l_new = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        l_old = tf.keras.losses.categorical_crossentropy(y_true_old, y_pred_old)
        loss = l_new + bct_alpha * l_old
        return loss

    return bct_loss
