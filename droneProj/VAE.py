import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pafy
import tensorflow as tf
import tensorflow.keras.backend as K

from keras.layers import Input, Flatten, Reshape, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import os
from keras.layers import BatchNormalization, Lambda
from keras.losses import binary_crossentropy
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape

# Data & model configuration
img_width, img_height = 128, 128
batch_size = 128
no_epochs = 100
latent_dim = 1
num_channels = 3
input_shape = (img_height, img_width, num_channels)
DIMS = (img_height, img_width, num_channels)


def min_max_norm(img):
    return (img - np.min(img)) / np.ptp(img)

# Define sampling with reparameterization trick
def sample_z(args):
    mu, sigma = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape=(batch, dim))
    return mu + K.exp(sigma / 2) * eps


# Define loss
def kl_reconstruction_loss(true, pred):
    beta = 6.0
    # Reconstruction loss
    reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * img_width * img_height
    # KL divergence loss
    kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    kl_loss *= beta
    # Total loss = 50% rec + 50% KL divergence loss
    return K.mean(reconstruction_loss + kl_loss)


def build_encoder(weights_name):
    # Define encoder's structure
    i = Input(shape=input_shape, name='encoder_input')
    cx = Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(i)
    cx = BatchNormalization()(cx)
    cx = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(cx)
    cx = BatchNormalization()(cx)
    x = Flatten()(cx)
    x = Dense(20, activation='relu')(x)
    x = BatchNormalization()(x)
    mu = Dense(latent_dim, name='latent_mu')(x)
    sigma = Dense(latent_dim, name='latent_sigma')(x)

    # Get Conv2D shape for Conv2DTranspose operation in decoder
    conv_shape = K.int_shape(cx)

    # Use reparameterization trick to ensure correct gradient
    z = Lambda(sample_z, output_shape=(latent_dim,), name='z')([mu, sigma])

    # Instantiate encoder
    encoder = Model(i, [mu, sigma, z], name=f'encoder_{weights_name}')
    return i, conv_shape, encoder


# =================
# Decoder
# =================
# Definition
def build_decoder(weights_name, conv_shape):
    d_i = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(d_i)
    x = BatchNormalization()(x)
    x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
    cx = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    cx = BatchNormalization()(cx)
    cx = Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(cx)
    cx = BatchNormalization()(cx)
    o = Conv2DTranspose(filters=num_channels, kernel_size=3, activation='sigmoid', padding='same',
                        name='decoder_output')(cx)

    # Instantiate decoder
    decoder = Model(d_i, o, name=f'decoder_{weights_name}')
    return decoder


# Instantiate VAE
def build_vae(weights_type):
    i, conv_shape, encoder = build_encoder(weights_type)
    decoder = build_decoder(weights_type, conv_shape)
    vae_outputs = decoder(encoder(i)[2])
    vae = Model(i, vae_outputs, name=f'vae_{weights_type}')
    return vae, encoder, decoder