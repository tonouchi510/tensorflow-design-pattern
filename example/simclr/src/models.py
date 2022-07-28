from tkinter import W
from typing import List
import tensorflow as tf


class SimCLRModel(tf.keras.Model):
    """分散学習でSimCLRを学習するためのモデル.

    論文: https://arxiv.org/abs/2002.05709
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_batch_size = kwargs["global_batch_size"]
        self.embedded_dim = kwargs["embedded_dim"]
    
    @tf.function
    def merge_fn(self, _, per_replica_res):
        return self.distribute_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_res, axis=None)

    def train_step(self, x1, x2):
        """Contrastive Learningのトレーニングステップ.

        引数 x1, x2は、同じ画像に対して異なるaugmentされた画像のペア
        """
        x = tf.concat([x1, x2], 0)

        with tf.GradientTape() as tape:
            z = self(x)
            z = tf.math.l2_normalize(z, -1)

            z1, z2 = tf.split(z, 2, 0)
            z = tf.concat([z1, z2], -1)

            replica_context = tf.distribute.get_replica_context()
            replica_id = replica_context.replica_id_in_sync_group
            num_replicas = replica_context.num_replicas_in_sync

            per_replica_res = tf.scatter_nd(
                indices=[[replica_id]],
                updates=[z],
                shape=[num_replicas] + [int(self.global_batch_size / num_replicas), self.embedded_dim * 2])

            z = tf.distribute.get_replica_context().merge_call(self.merge_fn, args=(per_replica_res,))
            z = tf.reshape(z, [-1] + z.shape.as_list()[2:])

            zis, zjs = tf.split(z, 2, -1)
            loss = self.compiled_loss(zis, zjs)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {m.name: m.result() for m in self.metrics}


def build_model(
    input_shape: List,
    global_batch_size: int,
    n_dim: int = 512,
    emb_dim: int = 512,
    checkpoint: str = "",
) -> SimCLRModel:
    base_model = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=input_shape)
    base_model.trainable = True

    inputs = tf.keras.layers.Input(input_shape)
    h = base_model(inputs, training=True)
    h = tf.keras.layers.GlobalAveragePooling2D()(h)

    projection_1 = tf.keras.layers.Dense(n_dim * 4)(h)
    projection_1 = tf.keras.layers.Activation("relu")(projection_1)
    projection_2 = tf.keras.layers.Dense(n_dim * 2)(projection_1)
    projection_2 = tf.keras.layers.Activation("relu")(projection_2)
    projection_3 = tf.keras.layers.Dense(n_dim)(projection_2)

    model = SimCLRModel(
        inputs, projection_3, global_batch_size=global_batch_size, embedded_dim=emb_dim)

    if checkpoint:
        model.load_weights(checkpoint)
    return model
