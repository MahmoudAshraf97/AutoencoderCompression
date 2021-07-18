
import tensorflow as tf
import argparse
from glob import glob
from itertools import product
import numpy as np
import os
from tqdm import tqdm
from scipy.special import erf
from modules import SSIM_MAE_VGG_Loss, PSNR, Bit_Loss, MaskConv2D
from skimage.io import imread

import lzma

physical_devices = tf.config.list_physical_devices('GPU')
try:
    for gpu_instance in physical_devices:

        tf.config.experimental.set_memory_growth(gpu_instance, True)
except:
    print('Invalid device or cannot modify virtual devices once initialized.')
    pass

MAX_N = 65536
TINY = 1e-10

##process the image into suitable dimension
def load_img(path):
    image = imread(path) / 255.


    image = image.astype(float)

    h, w, _ = image.shape
    h_, w_ = h, w
    if h % 16 != 0:
        h_ = (h // 16 + 1) * 16
    if w % 16 != 0:
        w_ = (w // 16 + 1) * 16
    img_ = np.zeros((1, h_, w_, 3))
    img_[:, :h, :w, :] = image
    return img_, h_ - h, w_ - w


def load_model(args):

    model = tf.keras.models.load_model(args.model_path,
                                       custom_objects={"SSIM_MAE_VGG_Loss": SSIM_MAE_VGG_Loss,
                                                       "Bit_Loss": Bit_Loss,
                                                       "PSNR": PSNR,
                                                       "MaskConv2D": MaskConv2D})
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

        img, pad_h, pad_w = load_img(path)
        _, H, W, _ = img.shape



        y_hat = model.layers[1](img)[0].numpy()




        # store side information
        #with open(bitpath, mode='wb') as fileobj:
            #img_size = np.array([W, H], dtype=np.uint16)
            #img_size.tofile(fileobj)
            #pad_size = np.array([pad_w, pad_h], dtype=np.uint8)
            #pad_size.tofile(fileobj)

        with lzma.open(bitpath, "wb", preset=9) as lzf:
            lzf.write(y_hat.flatten().astype(np.int8))
            lzf.write(np.array([W, H], dtype=np.uint16))
            lzf.write(np.array([pad_w, pad_h], dtype=np.uint8))
        # store side information
        #with open(bitpath, mode='ab') as fileobj:
            #img_size = np.array([W, H], dtype=np.uint16)
            #img_size.tofile(fileobj)
            #pad_size = np.array([pad_w, pad_h], dtype=np.uint16)
            #pad_size.tofile(fileobj)
        #fileobj.close()

        print('=============================================================')
        print(os.path.basename(path))

        real_bpp = os.path.getsize(bitpath) * 8
        print('bitrate : {0:.4}bpp'.format(real_bpp / H / W))
        print('=============================================================\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path',type=str, default='SSIM_MAE_VGG_COCO_model_none.h5')
    parser.add_argument('image_path',type=str, default='kodak/kodim20.png')

    parser.add_argument('--bottleneck', type=int, default=32)
    parser.add_argument('--main_channel', type=int, default=192)
    parser.add_argument('--gmm_K', type=int, default=3)

    args = parser.parse_args()
    compress(args)