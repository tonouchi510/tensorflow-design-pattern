import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from typing import List
from absl import flags

FLAGS = flags.FLAGS


def build_model(input_shape: List, num_classes: int):
    """トレーニングに使用するモデルを作成する.

    Args:
        input_shape {List} -- 入力データのshape.
        num_classes {int} -- クラス数.
    """
    inputs = tf.keras.Input(shape=input_shape)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(inputs)

    model = CustomModelClass(inputs, outputs)
    # カスタムモデルクラスをしようしない場合は以下
    #model = tf.keras.Model(inputs, outputs)
    return model


class CustomModelClass(tf.keras.Model):
    """トレーニングループをカスタマイズするための自前クラス.

    tf.keras.Model.fit時に内部で呼び出される`train_step`メソッドをオーバーライドすることで、
    トレーニングループのカスタマイズが可能.


    Args:
        tf ([type]): [description]
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # クラス内で参照が必要なパラメータを設定
        self.param_a = "~~~"
        self.param_b = "~~~"

    def train_step(self, x, y):
        """ここにトレーニングループの実装を書く.

        引数の数なども任意に設定できる.
        `tf.keras.Model.compile`で指定した損失関数やOptimizerにもアクセスできる.


        Args:
            x {Tensor} -- モデルへの入力（例えば入力画像）
            y {Tensor} -- モデルへの入力（例えば教師ラベル）

        Returns:
            [type]: [description]
        """

        # 何らかの処理
        with tf.GradientTape() as tape:
            # 何らかの処理
            y_pred = self(x, training=True)  # selfのみのcallはモデルへの入力
            # 何らかの処理
            loss = self.compiled_loss(y, y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {m.name: m.result() for m in self.metrics}