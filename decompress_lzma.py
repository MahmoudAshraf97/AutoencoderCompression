import argparse
import cv2
from collections import OrderedDict
from glob import glob
from itertools import product
import numpy as np
import os
from tqdm import tqdm
from scipy.special import erf
import tensorflow as tf
from modules import SSIM_MAE_VGG_Loss, PSNR, Bit_Loss, MaskConv2D
import lzma

MAX_N = 65536
TINY = 1e-10



def load_model(args):

    model = tf.keras.models.load_model(args.model_path,
                                       custom_objects={"SSIM_MAE_VGG_Loss": SSIM_MAE_VGG_Loss,
                                                       "Bit_Loss": Bit_Loss,
                                                       "PSNR": PSNR,
                                                       "MaskConv2D": MaskConv2D})
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

        with lzma.open(path, "rb") as lzf:
            buf = lzf.read()
            y_hat = np.frombuffer(buf,count=len(buf)-6,dtype=np.int8)

            arr = np.frombuffer(buf,offset=len(buf)-6, dtype=np.uint16)
            W, H = int(arr[0]), int(arr[1])

            arr = np.frombuffer(buf,offset=len(buf)-2, dtype=np.uint8)
            pad_w, pad_h = int(arr[0]), int(arr[1])


        y_hat = y_hat.reshape((1,H//16,W//16,32))

        fake_images = model.layers[3](y_hat)

        fake_images = fake_images[:, :H - pad_h, :W - pad_w, :]

        fakepath = "./outputs/reconstruction/{}.jpg".format(os.path.basename(path).split('.')[0])
        tf.keras.preprocessing.image.save_img(fakepath, tf.squeeze(fake_images))

        print('========================================================================\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('binary_path')

    parser.add_argument('--bottleneck', type=int, default=32)
    parser.add_argument('--main_channel', type=int, default=192)
    parser.add_argument('--gmm_K', type=int, default=3)

    args = parser.parse_args()
    comp_model = load_model(args)
    decompress(comp_model, args)