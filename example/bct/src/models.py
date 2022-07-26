import tensorflow as tf
from typing import List

# 古い特徴ベクトルでトレーニングされた分類器
OLD_CLASSIFIER = "gs://bucket-name/work_dir/classifier/saved_model"


class BCTModel(tf.keras.Model):
    """Backward-Compatible Trainingするためのカスタムモデルクラス.
    手法: https://arxiv.org/abs/2003.11942
    """

    def __init__(self, *args, **kwargs):
        super(BCTModel, self).__init__(*args, **kwargs)
        self.old_fc = tf.keras.models.load_model(OLD_CLASSIFIER)
        self.old_fc.trainable = False
        self.old_fc._name = "old_head"

    def compile(self, optimizer, loss):
        super(BCTModel, self).compile()
        self.loss_fn = loss
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.optimizer = optimizer
        # 新旧分類器それぞれにmetricsを用意
        self.fc_metrics = tf.keras.metrics.CategoricalAccuracy(name="acc")
        self.old_fc_metrics = tf.keras.metrics.CategoricalAccuracy()

    def train_step(self, data):
        """BCT用のトレーニングステップ.

        抽出した特徴ベクトルを旧分類器にも入力し、誤差を加算する.
        """
        x, y_true, y_true_old = data
        with tf.GradientTape() as tape:
            y_pred, emb = self(x, training=True)
            y_pred_old = self.old_fc(emb, training=False)
            loss = self.loss_fn(y_true, y_pred, y_true_old, y_pred_old)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.fc_metrics.update_state(y_true, y_pred)
        self.old_fc_metrics.update_state(y_true_old, y_pred_old)
        self.loss_tracker.update_state(loss)

        return {
            "loss": self.loss_tracker.result(),
            "head_acc": self.fc_metrics.result(),
            "old_head_acc": self.old_fc_metrics.result()
        }

    def test_step(self, data):
        x, y_true, y_true_old = data
        y_pred, emb = self(x, training=False)
        y_pred_old = self.old_fc(emb, training=False)
        loss = self.loss_fn(
            y_true, y_pred, y_pred_old, y_true_old)
        self.fc_metrics.update_state(y_true, y_pred)
        self.old_fc_metrics.update_state(y_true_old, y_pred_old)
        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
            "head_acc": self.fc_metrics.result(),
            "old_head_acc": self.old_fc_metrics.result()
        }


def build_model(
    input_shape: List,
    emb_dim: int,
    num_classes: int,
    pretrained: bool = True,
    checkpoint: str = "",
):
    """トレーニングに使用するモデルを作成する.

    シンプルかつ拡張性高く保つため、基本的に tf.keras.applications に実装されているモデルをベースに構築している.

    Args:
        input_shape {List} -- 入力データのshape.
        emb_dim {int} -- 埋め込みベクトル（特徴ベクトル）の次元数.
        num_classes {int} -- クラス数.
        pretrained {bool} -- ベースモデルの学習済みWeightを使用するかどうか.
        checkopint {str, optional} -- checkpointファイルがある場合に設定.
    """
    inputs = tf.keras.Input(shape=input_shape)
    base_model = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=False,
        input_shape=input_shape,
        weights="imagenet" if pretrained else None,
    )
    h = base_model(inputs, training=True)
    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    emb = tf.keras.layers.Dense(emb_dim, name="embedding_layer")(h)
    head = tf.keras.layers.Dense(num_classes, name="head", activation="softmax")(emb)

    # 埋め込みベクトルも出力するように構築
    model = BCTModel(inputs=inputs, outputs=[head, emb])

    if checkpoint:
        model.load_weights(checkpoint)
    return model


