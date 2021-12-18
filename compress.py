import os
import tensorflow as tf
import tensorflow_compression as tfc
import argparse
from glob import glob

##process the image into suitable dimension
def load_img(path):
    string = tf.io.read_file(path)
    image = tf.image.decode_image(string, channels=3)
    return image


def load_model(args):
    model = tf.keras.models.load_model(args.model_path,compile=False)
    return model


def compress(args):
    model = load_model(args)


    os.makedirs('outputs/binary', exist_ok=True)

    if os.path.isdir(args.image_path):
        pathes = glob(os.path.join(args.image_path, '*'))
    else:
        pathes = [args.image_path]

    for path in pathes:
        bitpath = "outputs/binary/{}.pth".format(os.path.basename(path).split('.')[0])

        image = load_img(path)
        compressed = model.compress(image)
        packed = tfc.PackedTensors()
        packed.pack(compressed)
        with open(bitpath, "wb") as f:
            f.write(packed.string)
        num_pixels = tf.reduce_prod(tf.shape(image)[:-1])
        bpp = len(packed.string) * 8 / num_pixels




        print('=============================================================')
        print(os.path.basename(path))

        print('bitrate : {0:.4}bpp'.format(bpp))
        print('=============================================================\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path',type=str, default='final_model')
    parser.add_argument('image_path',type=str, default='kodak/kodim20.png')

    args = parser.parse_args()

    compress(args)