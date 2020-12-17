import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.data.ops.readers import TFRecordDatasetV2
from tensorflow.python.framework.ops import Tensor
from typing import List
import json


def get_dataset(dataset_path: str,
                global_batch_size: int,
                split: str) -> TFRecordDatasetV2:
    """tensorflowのデータパイプラインの構築を行う関数.

    ここではTFRecordを使う実装例を紹介している.
    必要に応じて任意にカスタマイズして使う.

    Args:
        dataset_path {str} -- データセットファイルが保存されているパス.
        split {str} -- データセットのスプリット（train/valid/test）
        global_batch_size {int} -- バッチサイズ. 特に、distributeStrategyを使う前は分割前のバッチサイズ. 

    Returns:
        TFRecordDatasetV2:  構築したtensorflowのデータパイプライン.
    """
    def preprocessing(example: Tensor, param: str):
        """データパイプラインに組み込む前処理関数

        tfrecordのパースや入力画像の前処理、リサイズ等々.

        Args:
            example (Tensor): 
            param (str): 前処理関数に渡す何らかのパラメータ

        Returns:
            [type]: [description]
        """
        return

    file_names = tf.io.gfile.glob(f"{dataset_path}/{split}-*.tfrec")

    # Build a pipeline
    option = tf.data.Options()
    option.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(file_names, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    if split == "train":
        dataset = (
            dataset
                .with_options(option)
                .map(lambda x: preprocessing(x, "~~~"), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .shuffle(512, reshuffle_each_iteration=True)
                .batch(global_batch_size, drop_remainder=True)
                .prefetch(tf.data.experimental.AUTOTUNE)
        )
    else:
        dataset = (
            dataset
                .with_options(option)
                .map(lambda x: preprocessing(x, "~~~"), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .batch(global_batch_size, drop_remainder=False)
                .prefetch(tf.data.experimental.AUTOTUNE)
        )
    return dataset