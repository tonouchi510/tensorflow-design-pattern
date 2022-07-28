from typing import Callable
import tensorflow as tf


# データパイプラインの構築用関数
def create_data_pipeline(
    dataset_path: str,
    batch_size: int,
    split: str,
    preprocess_func: Callable,
    augmentation_func: Callable = lambda x, y: (x, y)
):
    """tensorflowのデータパイプラインの構築を行う関数.

    ここではTFRecordを使う実装例を紹介している.必要に応じて任意にカスタマイズして使う.

    Args:
        dataset_path {str} -- 目的のTFRecordファイルが保存されているパス.
        batch_size {int} -- バッチサイズ(分散処理の場合は合計).
        split {str} -- train, valid or test.
        preprocess_func {Callable} -- 適用する前処理関数.
        augmentation_func {Callable, optional}: データオーグメンテーション関数. Defaults to lambdax:x.

    Returns:
        構築したtensorflowのデータパイプライン.
    """
    file_names = tf.io.gfile.glob(f"{dataset_path}/{split}-*.tfrec")

    # Build a pipeline
    option = tf.data.Options()
    option.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(file_names, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    if split == "train":
        dataset = (
            dataset
                .with_options(option)
                .map(lambda example: preprocess_func(example), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .map(lambda x, y=None: augmentation_func(x, y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .shuffle(512, reshuffle_each_iteration=True)
                .batch(batch_size, drop_remainder=True)
                .prefetch(tf.data.experimental.AUTOTUNE)
        )
    elif split == "valid" or split == "test":
        # valid/testデータの時はDAやshuffleしない
        dataset = (
            dataset
                .with_options(option)
                .map(lambda example: preprocess_func(example), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .batch(batch_size, drop_remainder=False)
                .prefetch(tf.data.experimental.AUTOTUNE)
        )
    else:
        raise ValueError("train/valid/testの中から選んでください。")
    return dataset
