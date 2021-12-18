import numpy as np
from tensorflow_compression import GDN

import tensorflow as tf
import tensorflow_compression as tfc
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv2D, Add, Lambda, Multiply, Softmax
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.applications import VGG19
from LPIPS import LPIPSLoss


def convolutional_block(X, out_ch, downsample, actv2):
    """
    Implementation of the Residual convolutional block
    Skip Connections:
    They mitigate the problem of vanishing gradient by allowing the alternate shortcut path for gradient to flow through
    They allow the model to learn an identity function which ensures that the higher layer will perform at least as good as the lower layer, and not worse


    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    out_ch -- Number of filters
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)

    https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33#43fa
    """

    stride = 2 if downsample else 1

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(out_ch, kernel_size=3, strides=stride, padding='same',
               activation=LeakyReLU(alpha=0.2))(X)

    # Second component of main path
    if actv2 == 2:
        X = Conv2D(out_ch, kernel_size=3, padding='same',
                   activation=GDN(inverse=True))(X)
    if actv2 == 1:
        X = Conv2D(out_ch, kernel_size=3, padding='same',
                   activation=GDN(inverse=False))(X)
    else:
        X = Conv2D(out_ch, kernel_size=3, padding='same')(X)

    ##### SHORTCUT PATH #### (≈2 lines) if downsample handles different channels, replace by identity if not needed
    # if downsample:
    X_shortcut = Conv2D(out_ch, kernel_size=(1, 1), strides=stride, padding='same')(X_shortcut)

    # Final step: Add shortcut value to main path
    X = Add()([X, X_shortcut])

    return X


def NLAM(X, out_ch, downsample):
    """
    Non-Local Attention Module
    Attention modules are used to make CNN learn and focus more on the important information,
    rather than learning non-useful background information. In the case of object detection,
    useful information is the objects or target class crop that we want to classify and localize in an image.
    The attention module consists of a simple 2D-convolutional layer, MLP(in the case of channel attention),
    and sigmoid function at the end to generate a mask of the input feature map.

    """

    # Internal Parameters
    actv2 = 1

    stride = 2 if downsample else 1

    # Save the input value
    X_shortcut = X
    X_main = X

    # Mask Block
    X = convolutional_block(X, out_ch, downsample, actv2)
    X = convolutional_block(X, out_ch, downsample, actv2)
    X = convolutional_block(X, out_ch, downsample, actv2)
    # Conv Layer
    X = Conv2D(out_ch, kernel_size=1, strides=1, padding='same', activation='sigmoid')(X)


    # Main Block
    X_main = convolutional_block(X_main, out_ch, downsample, actv2)
    X_main = convolutional_block(X_main, out_ch, downsample, actv2)
    X_main = convolutional_block(X_main, out_ch, downsample, actv2)

    # Final
    # if downsample:
    X_shortcut = Conv2D(out_ch, kernel_size=(1, 1), strides=stride ** 3, padding='same')(X_shortcut)
    Y = Multiply()([X, X_main])
    Y = Add()([Y, X_shortcut])

    return Y


def UpSample(X, out_ch):
    # Save the input value
    X_shortcut = X

    # Path 1
    X = Conv2D(4 * out_ch, kernel_size=3, padding='same', activation=LeakyReLU(alpha=0.2))(X)
    X = Lambda(lambda x: tf.nn.depth_to_space(x, 2))(X)
    X = Conv2D(out_ch, kernel_size=3, padding='same', activation=GDN(inverse=True))(X)

    # Path 2
    X_shortcut = Conv2D(4 * out_ch, kernel_size=1, padding='same')(X_shortcut)

    X_shortcut = Lambda(lambda x: tf.nn.depth_to_space(x, 2))(X_shortcut)

    # Final step: Add shortcut value to main path
    Y = Add()([X, X_shortcut])

    return Y

ch = 192
out_ch = 32

##Encoder

img_input = Input(shape=(None, None, 3), name='Image')

ResBlock_1 = convolutional_block(img_input, ch, 1, 1)
ResBlock_2 = convolutional_block(ResBlock_1, ch, 0, 1)
ResBlock_3 = convolutional_block(ResBlock_2, ch, 1, 1)
NLAM_1 = NLAM(ResBlock_3, ch, 0)
ResBlock_4 = convolutional_block(NLAM_1, ch, 0, 1)
ResBlock_5 = convolutional_block(ResBlock_4, ch, 1, 1)
ResBlock_6 = convolutional_block(ResBlock_5, ch, 0, 1)
NLAM_2 = NLAM(ResBlock_6, ch, 0)
final_conv2d = Conv2D(out_ch, kernel_size=3, strides=2, padding='same',
                      activation=LeakyReLU(alpha=0.2))(NLAM_2)

Encoder = Model(inputs=img_input, outputs=final_conv2d, name='Encoder')

##Decoder
decoder_input = Input(shape=(None, None, out_ch), name='Decoder_Input')
conv2d_1 = Conv2D(ch, kernel_size=3, padding='same',
                  activation=LeakyReLU(alpha=0.2))(decoder_input)
NLAM_1 = NLAM(conv2d_1, ch, 0)
ResBlock_1 = convolutional_block(NLAM_1, ch, 0, 2)
UpResBlock_1 = UpSample(ResBlock_1, ch)
ResBlock_2 = convolutional_block(UpResBlock_1, ch, 0, 2)
UpResBlock_2 = UpSample(ResBlock_2, ch)
NLAM_2 = NLAM(UpResBlock_2, ch, 0)
ResBlock_3 = convolutional_block(NLAM_2, ch, 0, 2)
UpResBlock_3 = UpSample(ResBlock_3, ch)
ResBlock_4 = convolutional_block(UpResBlock_3, ch, 0, 2)
UpResBlock_4 = UpSample(ResBlock_4, 3)

Decoder = Model(inputs=decoder_input, outputs=UpResBlock_4, name='Decoder')


class AutoencoderModel(tf.keras.Model):
    """Main model class."""

    def __init__(self, lmbda):
        super().__init__()

        self.lmbda = lmbda
        self.analysis_transform = Encoder
        self.synthesis_transform = Decoder
        self.prior = tfc.NoisyDeepFactorized(batch_shape=(out_ch,))
        self.build((None, None, None, 3))


    def call(self, x, training):
        """Computes rate and distortion losses."""
        entropy_model = tfc.ContinuousBatchedEntropyModel(self.prior, coding_rank=3, compression=False)

        y = self.analysis_transform(x)
        y_hat, bits = entropy_model(y, training=training)
        x_hat = self.synthesis_transform(y_hat)
        # Total number of bits divided by total number of pixels.
        num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), bits.dtype)
        bpp = tf.reduce_sum(bits) / num_pixels
        # Mean squared error across pixels.
        lpips_loss = LPIPSLoss("lpips_weights/net-lin_alex_v0.1.pb")

        mse = tf.reduce_mean(tf.abs(tf.cast(x,x_hat.dtype)-x_hat)) + lpips_loss(tf.cast(x,x_hat.dtype),x_hat)


        # The rate-distortion Lagrangian.
        loss = bpp + self.lmbda * mse
        psnr = tf.reduce_mean(tf.image.psnr(tf.cast(x,x_hat.dtype), x_hat, 1.0))
        return loss, bpp, mse, psnr


    def train_step(self, x):

        with tf.GradientTape() as tape:
            loss, bpp, mse, psnr = self(x, training=True)
        variables = self.trainable_variables
        self.optimizer.minimize(loss, variables,tape=tape)

        self.loss.update_state(loss)
        self.bpp.update_state(bpp)
        self.mse.update_state(mse)
        self.psnr.update_state(psnr)
        return {m.name: m.result() for m in [self.loss, self.bpp, self.mse, self.psnr]}

    def test_step(self, x):
        loss, bpp, mse, psnr = self(x, training=False)
        self.loss.update_state(loss)
        self.bpp.update_state(bpp)
        self.mse.update_state(mse)
        self.psnr.update_state(psnr)
        return {m.name: m.result() for m in [self.loss, self.bpp, self.mse, self.psnr]}

    def predict_step(self, x):
        raise NotImplementedError("Prediction API is not supported.")

    def compile(self, **kwargs):
        super().compile(
            loss=None,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            **kwargs,
        )
        self.loss = tf.keras.metrics.Mean(name="loss")
        self.bpp = tf.keras.metrics.Mean(name="bpp")
        self.mse = tf.keras.metrics.Mean(name="PerceptualLoss")
        self.psnr = tf.keras.metrics.Mean(name="PSNR")

    def fit(self, *args, **kwargs):
        retval = super().fit(*args, **kwargs)
        # After training, fix range coding tables.
        self.entropy_model = tfc.ContinuousBatchedEntropyModel(self.prior, coding_rank=3, compression=True)
        return retval

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
    ])
    def compress(self, x):
        """Compresses an image."""
        #Zero Pad to multiples of 16
        h, w, _ = x.shape
        if h % 16 != 0:
            h_ = (h // 16 + 1) * 16
        if w % 16 != 0:
            w_ = (w // 16 + 1) * 16
        paddings = tf.constant([[0, h_ - h], [0, w_ - w], [0, 0]])
        x_padded = tf.pad(x, paddings, "CONSTANT")

        # Add batch dimension and cast to float.
        x_padded = tf.expand_dims(x_padded, 0)
        x_padded = tf.cast(x_padded, dtype=tf.float32)
        x_padded = x_padded / 255
        y = self.analysis_transform(x_padded)
        # Preserve spatial shapes of both image and latents.
        x_shape = tf.shape(x)[1:-1]
        y_shape = tf.shape(y)[1:-1]
        return self.entropy_model.compress(y), x_shape, y_shape

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(1,), dtype=tf.string),
        tf.TensorSpec(shape=(2,), dtype=tf.int32),
        tf.TensorSpec(shape=(2,), dtype=tf.int32),
    ])
    def decompress(self, string, x_shape, y_shape):
        """Decompresses an image."""
        y_hat = self.entropy_model.decompress(string, y_shape)
        x_hat = self.synthesis_transform(y_hat)
        # Remove batch dimension, and crop away any extraneous padding.
        x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
        # Then cast back to 8-bit integer.
        x_hat = x_hat * 255
        return tf.saturate_cast(tf.round(x_hat), tf.uint8)
