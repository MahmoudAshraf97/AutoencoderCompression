import tensorflow as tf
from tensorflow_addons.optimizers import CyclicalLearningRate
from modules import AutoencoderModel

import os
import random
from glob import glob
from pathlib import Path


#tf.keras.mixed_precision.set_global_policy('mixed_float16')

BATCH_SIZE = 4
HEIGHT = 256
WIDTH = 256

##########################
def get_dataset(filenames, batch_size):
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
        .map(prepare_sample, num_parallel_calls=AUTOTUNE)
        .shuffle(batch_size * 10)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return dataset
def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    return example
def prepare_sample(features):
    image = tf.image.resize(tf.cast(features["image"],dtype=tf.float32)/255., size=(HEIGHT, WIDTH))
    return image
tfrecords_dir = "coco_tfrecords"
TRAIN_DIR = os.path.join(tfrecords_dir, "train")
VAL_DIR = os.path.join(tfrecords_dir, "valid")
train_filenames = tf.io.gfile.glob(os.path.join(TRAIN_DIR, "*.tfrec"))
validation_filenames = tf.io.gfile.glob(os.path.join(VAL_DIR, "*.tfrec"))


AUTOTUNE = tf.data.AUTOTUNE
##########################################


model = AutoencoderModel(1)

INIT_LR = 5e-5
MAX_LR = 1e-4
clr = CyclicalLearningRate(initial_learning_rate=INIT_LR,
                           maximal_learning_rate=MAX_LR,
                           scale_fn=lambda x: 1/(2.**(x-1)),
                           step_size=2 * 29572)

opt = tf.keras.optimizers.Adam(learning_rate=clr)
#opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
model.compile(optimizer=opt)
#model.load_weights("final_model")
chk_point = tf.keras.callbacks.ModelCheckpoint(filepath='model/model.{epoch:02d}-{val_loss:.2f}.h5',
                                               save_best_only=True,
                                               save_weights_only=True)

model.fit(x=get_dataset(train_filenames,batch_size=BATCH_SIZE),
            validation_data=get_dataset(validation_filenames, batch_size=BATCH_SIZE),
            epochs=10,
            callbacks = chk_point,
            verbose=int(True)
            )

model.save('final_model')
