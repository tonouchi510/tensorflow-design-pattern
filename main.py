import tensorflow as tf
import numpy as np
from absl import app
from absl import flags

from utils import create_data_pipeline, get_callbacks
from src.data import preprocess_func_partial, augmentation_func_partial
from src.losses import loss_func
from src.models import build_model

# Random seed fixation
tf.random.set_seed(666)
np.random.seed(666)

INPUT_SHAPE = [112, 112, 3]

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "work_dir", "gs://backet-name/experiments/experiment_name",
    "GCS path for training job management.")

flags.DEFINE_string(
    "dataset_path", "gs://bucket-name/datasets/dataset_name",
    "Directory where tfrecord dataset is stored.")

flags.DEFINE_string(
    "model_type", "efficientnet",
    "Parameter for selecting the base model structure.")

flags.DEFINE_integer(
    "epochs", 100,
    "Number of epochs to train.")

flags.DEFINE_integer(
    "last_epoch", 0,
    "Set when recovering from checkpoint.")

flags.DEFINE_integer(
    "global_batch_size", 2048,
    "Batch size for training/eval before distribution.")

flags.DEFINE_float(
    "learning_rate", 0.001,
    "Initial learning rate")

flags.DEFINE_integer(
    "num_classes", 0,
    "Num of class label.")

flags.DEFINE_bool(
    "use_tpu", False,
    "Whether to use tpu strategy.")


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # モデルの構築
    if FLAGS.use_tpu:
        # tf.distribute.Strategyを使う場合はこちら(TPU Strategyの例)
        cluster = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(cluster)
        tf.tpu.experimental.initialize_tpu_system(cluster)
        distribute_strategy = tf.distribute.TPUStrategy(cluster)
        with distribute_strategy.scope():
            # distribute_strategyのスコープ内でbuildする必要がある
            model = build_model(FLAGS.model_type, INPUT_SHAPE, num_classes=FLAGS.num_classes)
    else:
        model = build_model(FLAGS.model_type, INPUT_SHAPE, num_classes=FLAGS.num_classes)
    model.summary()

    # トレーニングの設定
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    model.compile(loss=loss_func, optimizer=optimizer, metrics=["accuracy"])

    # データパイプライン構築
    train_ds = create_data_pipeline(
        dataset_path=FLAGS.dataset_path,
        batch_size=FLAGS.global_batch_size,
        split="train",
        preprocess_func=preprocess_func_partial,
        augmentation_func=augmentation_func_partial
    )
    valid_ds = create_data_pipeline(
        dataset_path=FLAGS.dataset_path,
        batch_size=FLAGS.global_batch_size,
        split="valid",
        preprocess_func=preprocess_func_partial
    )

    # コールバックの取得
    callbacks = get_callbacks(FLAGS.work_dir)
    # トレーニング実行
    model.fit(
        train_ds,
        validation_data=valid_ds,
        callbacks=callbacks,
        initial_epoch=FLAGS.last_epoch,
        epochs=FLAGS.epochs
    )
    # 学習済みモデルを保存
    model.save(f"{FLAGS.job_dir}/saved_model", include_optimizer=False)


if __name__ == "__main__":
    app.run(main)
