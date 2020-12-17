from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.python.framework.ops import Tensor
import tensorflow as tf
import numpy as np
from absl import app
from absl import flags

from data import get_dataset
from models import build_model
from losses import custom_loss_func

# Random seed fixation
tf.random.set_seed(666)
np.random.seed(666)

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'global_batch_size', 2048,
    'Batch size for training/eval before distribution.')

flags.DEFINE_integer(
    'epochs', 100,
    'Number of epochs to train.')

flags.DEFINE_float(
    'learning_rate', 0.001,
    'Initial learning rate')

flags.DEFINE_string(
    'dataset', None,
    'Directory where dataset is stored.')

flags.DEFINE_string(
    'job_dir', None,
    'GCS path for training job management.')

flags.DEFINE_string(
    'num_classes', None,
    'Num of class label.')

flags.DEFINE_string(
    'use_tpu', False,
    'Whether to use tpu strategy.')

flags.DEFINE_string(
    'use_gpu', False,
    'Whether to use gpu (mirrored strategy).')


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    if FLAGS.use_tpu:
        # tf.distribute.Strategyを使う場合はこちらに設定
        # ここではtpu strategyの例
        cluster = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(cluster)
        tf.tpu.experimental.initialize_tpu_system(cluster)
        distribute_strategy = tf.distribute.TPUStrategy(cluster)

        with distribute_strategy.scope():
            model = build_model(FLAGS.input_shape, num_classes=FLAGS.num_classes)
            optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    elif FLAGS.use_gpu:
            # Setup mirrored strategy
            distribute_strategy = tf.distribute.MirroredStrategy()
            with distribute_strategy.scope():
                model = build_model(FLAGS.input_shape, num_classes=FLAGS.num_classes)
                optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    else:
        model = build_model(FLAGS.input_shape, num_classes=FLAGS.num_classes)
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    model.compile(loss=custom_loss_func,
                  optimizer=optimizer,
                  metrics=["accuracy"])
    model.summary()

    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"{FLAGS.job_dir}/logs", histogram_freq=1)
    callbacks = [tboard_callback]

    train_ds = get_dataset(FLAGS.dataset, FLAGS.global_batch_size, "train")
    valid_ds = get_dataset(FLAGS.dataset, FLAGS.global_batch_size, "valid")

    for epoch in range(FLAGS.epochs):
        model.fit(train_ds, validation_data=valid_ds, callbacks=callbacks, initial_epoch=epoch, epochs=epoch+1)
        model.save(f"{FLAGS.job_dir}/checkpoints/{epoch+1}", include_optimizer=True)
    
    model.save(f"{FLAGS.job_dir}/saved_model", include_optimizer=False)


if __name__ == '__main__':
    app.run(main)
