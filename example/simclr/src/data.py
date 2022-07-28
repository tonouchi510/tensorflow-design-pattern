import tensorflow as tf
import functools
from absl import flags

FLAGS = flags.Flag


def preprocess_func(example, size: int):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)
    image = tf.image.decode_png(example['image'], channels=3)
    image = tf.image.resize_with_pad(image, size, size, method="bilinear", antialias=False)
    return image


def augmentation_func(image, y):
    """Contrastive Learning 用のオーグメンテーション関数

    同じ画像に対し、異なるaugmentをかけた画像ペアを返す
    ※第二引数は使わないが、便宜上
    """
    def _color_jitter(x, s=1):
        # one can also shuffle the order of following augmentations
        # each time they are applied.
        x = tf.image.random_brightness(x, max_delta=0.8*s)
        x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_hue(x, max_delta=0.2*s)
        x = tf.clip_by_value(x, 0, 1)
        return x

    def _color_drop(x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.tile(x, [1, 1, 1, 3])
        return x

    def _random_apply(func, x, p):
        return tf.cond(
            tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                    tf.cast(p, tf.float32)),
            lambda: func(x),
            lambda: x
        )

    x1 = _random_apply(tf.image.flip_left_right, image, p=0.5)
    # Randomly apply transformation (color distortions) with probability p.
    x1 = _random_apply(_color_jitter, x1, p=0.8)
    x1 = _random_apply(_color_drop, x1, p=0.2)

    x2 = _random_apply(tf.image.flip_left_right, image, p=0.5)
    x2 = _random_apply(_color_jitter, x2, p=0.8)
    x2 = _random_apply(_color_drop, x2, p=0.2)
    return x1, x2


preprocess_func_partial = functools.partial(preprocess_func, size=112)

augmentation_func_partial = functools.partial(augmentation_func)
