import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Input, Lambda
import tensorflow.keras.backend as K
from keras.layers.advanced_activations import LeakyReLU
import os
import random
from glob import glob
from pathlib import Path
from modules import convolutional_block, NLAM, UpSample, Quantize, SSIM_MAE_VGG_Loss, PSNR, bitcost, Parameter_Estimate, Bit_Loss

##Encoder

ch = 192
out_ch = 32
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

y_hard, y_soft = Quantize(final_conv2d)

Encoder = Model(inputs=img_input, outputs=[y_hard, y_soft], name='Encoder')

## Entropy Model
latent_code = Input(shape=(None, None, out_ch),name='Latent_Code')
params = Parameter_Estimate(latent_code,3)
bits_per_pixel = bitcost(params,latent_code)
Entropy_Model=Model(inputs=latent_code,outputs=[bits_per_pixel,params],name='Entropy_Model')


##Decoder
decoder_input = Input(shape=(None, None, out_ch),name='Decoder_Input')
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
conv2d_2 = Conv2D(12, kernel_size=3, padding='same')(ResBlock_4)

UpResBlock_4 = UpSample(conv2d_2, 3)

Decoder = Model(inputs=decoder_input, outputs = UpResBlock_4,name='Decoder')


##End to End
autoencoder_input = Input(shape=(None, None, 3), name='Image')
encoder_output = Encoder(autoencoder_input)[1]
bpp = Entropy_Model(encoder_output)[0]

decoder_output = Decoder(encoder_output)

autoencoder = Model(inputs=autoencoder_input,
                    outputs=[bpp, decoder_output],
                    name='End_to_End')

autoencoder.build((None, None, 3))

autoencoder.load_weights('model.h5')
autoencoder.save('model.h5',save_format="h5")
print('####################################################################')