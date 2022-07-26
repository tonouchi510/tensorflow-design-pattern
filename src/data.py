# 前処理や、データオーグメンテージョンに使用する関数

import tensorflow as tf
import numpy as np


# 前処理用関数（以下、画像分類の例）
def preprocess_func(
    example,
    size: int,
    num_classes: int
):
    """前処理をここに記述し、データパイプラインに渡す

    TFRecordをパースし、前処理にかけたデータを返す
    """
    # decode the TFRecord
    features = {
        "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "label": tf.io.FixedLenFeature([], tf.int32),
    }

    example = tf.io.parse_single_example(example, features)
    image = tf.image.decode_png(example["image"], channels=3)
    image = tf.image.resize_with_pad(image, size, size, method="bilinear", antialias=False)
    image = tf.image.per_image_standardization(image)
    label = tf.one_hot(example["label"], num_classes)
    return image, label


# データオーグメンテーション関数（以下、画像の基本的なオーグメンテージョン）
def augmentation_func(x, y):
    """データオーグメンテーションはここに記述
    Args:
        x: image
        y: label
    
    ※引数は学習データによって異なるので目的のものに書き換える
    """
    def augment_image(image):
        rand_r = np.random.random()
        h, w, c = image.get_shape()
        dn = np.random.randint(15, size=1)[0]+1
        if  rand_r < 0.25:
            image = tf.image.random_crop(image, size=[h-dn, w-dn, c])
            image = tf.image.resize(image, size=[h, w])
        elif rand_r >= 0.25 and rand_r < 0.75:
            image = tf.image.resize_with_crop_or_pad(image, h+dn, w+dn)
            image = tf.image.random_crop(image, size=[h, w, c])
        return image

    return augment_image(x), y
