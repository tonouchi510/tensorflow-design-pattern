from typing import List
import tensorflow as tf
from google.cloud import storage


class CheckpointCallBack(tf.keras.callbacks.Callback):
    """CheckpointをGCSに保存するコールバック.

    公式の ModelCheckpoint callback は、weightファイルをローカルにしか保存できないため、カスタムのコールバックを用意.
    """

    def __init__(self, checkpoint_dir: str, bucket_name: str):
        super(CheckpointCallBack, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.bucket = storage.Client().bucket(bucket_name)

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(f"{epoch:05d}.h5")
        blob = self.bucket.blob(f"{self.checkpoint_dir}/{epoch:05d}.h5")
        blob.upload_from_filename(f"{epoch:05d}.h5")


def get_callbacks(work_dir: str) -> List:
    """汎用的に使われるcallbacksをまとめて返す.

    Args:
        work_dir (str): ワーキングディレクトリ

    Returns:
        List: コールバックのリスト
    """
    # checkpoint
    s = work_dir.replace("gs://", "").split("/")
    bucket_name = s[0]
    path = "/".join(s[1:])
    model_checkpoint_callback = CheckpointCallBack(checkpoint_dir=f"{path}/checkpoints", bucket_name=bucket_name)
    # tensorboard
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"{work_dir}/logs", histogram_freq=1)
    return [tboard_callback, model_checkpoint_callback]
