import tensorflow as tf
from typing import List


# カスタムのモデルクラスの定義
class CustomModelClass(tf.keras.Model):
    """カスタムモデルクラス.

    tf.keras.Model のサブクラスを構築する。内部メソッドをオーバーライドするため。
    個人的には主に、トレーニングループのカスタマイズをするために使うことが多い.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # クラス内で参照が必要なパラメータを設定
        self.param_a = "~~~"
        self.param_b = "~~~"

    def train_step(self, data):
        """トレーニングループのオーバーライド.

        引数も任意に設定できる. 以下参照.
        https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit

        """
        # デフォ実装を示している。ここを任意のトレーニングステップにカスタマイズする.
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


# モデルの構築用関数
def build_model(
    model_type: str,
    input_shape: List,
    num_classes: int,
    pretrained: bool = True
):
    """トレーニングに使用するモデルを作成する.

    シンプルかつ拡張性高く保つため、基本的に tf.keras.applications に実装されているモデルをベースに構築している.

    Args:
        model_type {str} -- ベースのモデル構造を選択するためのパラメータ.
        input_shape {List} -- 入力データのshape.
        num_classes {int} -- クラス数.
    """
    if model_type == "efficientnet":
        base_model = tf.keras.applications.efficientnet.EfficientNetB0(
            include_top=False,
            input_shape=input_shape,
            weights="imagenet" if pretrained else None,
        )
    elif model_type == "resnet_v2":
        base_model = tf.keras.applications.resnet_v2.ResNet50V2(
            include_top=False,
            input_shape=input_shape,
            weights="imagenet" if pretrained else None,
        )
    else:
        raise ValueError(f"Invalid argument '{model_type}'.")

    inputs = tf.keras.Input(shape=input_shape)
    h = base_model(inputs, training=True)
    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(h)

    model = CustomModelClass(inputs, outputs)
    # カスタムモデルクラスを使用しない場合は普通に以下で構築
    #model = tf.keras.Model(inputs, outputs)
    return model


