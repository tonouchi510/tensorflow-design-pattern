import tensorflow as tf
import numpy as np
import functools
from absl import flags

FLAGS = flags.FLAGS

cosine_sim_1d = tf.keras.losses.CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)
cosine_sim_2d = tf.keras.losses.CosineSimilarity(axis=2, reduction=tf.keras.losses.Reduction.NONE)
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)


def simclr_loss_func(zis, zjs, global_batch_size, temperature):
    """SimCLRの損失関数.

    Args:
        zis (_type_): _description_
        zjs (_type_): _description_
        global_batch_size (_type_): _description_
        temperature (_type_): _description_
    """
    def _get_negative_mask(batch_size: int):
        # return a mask that removes the similarity score of equal/similar images.
        # this function ensures that only distinct pair of images get their similarity scores
        # passed as negative examples
        negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0
        return tf.constant(negative_mask)

    def _dot_simililarity_dim1(x, y):
        # x shape: (N, 1, C)
        # y shape: (N, C, 1)
        # v shape: (N, 1, 1)
        v = tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2))
        return v

    def _dot_simililarity_dim2(x, y):
        v = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    negative_mask = _get_negative_mask(global_batch_size)
    l_pos = _dot_simililarity_dim1(zis, zjs)
    l_pos = tf.reshape(l_pos, (global_batch_size, 1))
    l_pos /= temperature

    negatives = tf.concat([zjs, zis], axis=0)

    loss = 0
    for positives in [zis, zjs]:
        labels = tf.zeros(global_batch_size, dtype=tf.int32)

        l_neg = _dot_simililarity_dim2(positives, negatives)
        l_neg = tf.boolean_mask(l_neg, negative_mask)
        l_neg = tf.reshape(l_neg, (global_batch_size, -1))
        l_neg /= temperature

        logits = tf.concat([l_pos, l_neg], axis=1)
        loss += criterion(y_pred=logits, y_true=labels)
    loss = loss / (2 * global_batch_size)
    return loss


loss_func = functools.partial(
    simclr_loss_func,
    global_batch_size=FLAGS.global_batch_size,
    temperature=FLAGS.temperature
)
