import os
import tensorflow as tf
import tensorflow_compression as tfc
import argparse
from glob import glob

def load_model(args):
    model = tf.keras.models.load_model(args.model_path,compile=False)
    return model

def decompress(model, args):
    os.makedirs("outputs/reconstruction/", exist_ok=True)

    if os.path.isdir(args.binary_path):
        pathes = glob(os.path.join(args.binary_path, '*'))
    else:
        pathes = [args.binary_path]

    for path in pathes:

        print('========================================================================')
        print('image', os.path.basename(path))

        with open(path, "rb") as f:
            packed = tfc.PackedTensors(f.read())
        tensors = packed.unpack(dtypes)
        x_hat = model.decompress(*tensors)


        fakepath = "./outputs/reconstruction/{}.png".format(os.path.basename(path).split('.')[0])
        string = tf.image.encode_png(x_hat)
        tf.io.write_file(fakepath, string)


        print('========================================================================\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('binary_path')


    args = parser.parse_args()

    model = load_model(args)
    dtypes = [t.dtype for t in model.decompress.input_signature]

    decompress(model, args)