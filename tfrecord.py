import os
from glob import glob
import tensorflow as tf





def image_feature(value):
    """
    #Returns a bytes_list from a string / byte.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.image.encode_jpeg(value).numpy()]))

def create_example(image):
    feature = {
        "image": image_feature(image),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def create_record(img_list,num_samples,tfrecords_dir):

    if not img_list:
        print("Please provide a valid image list")
    if not os.path.exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)  # creating TFRecords output folder
    num_tfrecords = (len(img_list) // num_samples) if (len(img_list) % num_samples == 0) else (len(img_list) // num_samples) + 1

    for tfrec_num in range(num_tfrecords):
        samples = img_list[
                  (tfrec_num * num_samples): min(((tfrec_num + 1) * num_samples), (len(img_list) - 1))]
        with tf.io.TFRecordWriter(tfrecords_dir + "/file_%.2i-%i.tfrec" % (
        tfrec_num, len(samples))) as writer:

            print("file_%.2i-%i.tfrec" % (tfrec_num, len(samples)))

            for sample in samples:
                image = tf.io.decode_image(tf.io.read_file(sample),channels=3)
                example = create_example(image)
                writer.write(example.SerializeToString())


DATASET_DIR = os.path.join("tiny-imagenet-200", "")
TRAIN_DIR = os.path.join(DATASET_DIR, "TRAIN")
VAL_DIR = os.path.join(DATASET_DIR, "valid")
tfrecords_dir = "tiny_tfrecords"
training_data = [y for x in os.walk(TRAIN_DIR) for y in glob(os.path.join(x[0], '*JPEG'))]
validation_data = [y for x in os.walk(VAL_DIR) for y in glob(os.path.join(x[0], '*JPEG'))]

create_record(training_data,130000,os.path.join(tfrecords_dir, "train"))
create_record(validation_data,1300000,os.path.join(tfrecords_dir, "valid"))




##################################################################################################################
