import tensorflow as tf 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Input, Reshape, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import ZeroPadding2D, DepthwiseConv2D, Dense, Flatten, LeakyReLU, MaxPooling2D
from tensorflow.keras import regularizers
from hyperparams import Hyperparams
from tensorflow.keras.layers import LSTM
from rbf import RBFLayer
import logging

H = Hyperparams()
logger configuration
FORMAT = "[%(filename)s: %(lineno)3s] %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


class model(object):

    def __init__(self):
        return None

    def get_model_cnn(self):
        
        inp = Input((240, 360, 1))
        conv_layer = Conv2D(32, kernel_size=3, activation='relu', data_format='channels_last', padding='same')(inp)
        conv_layer = Conv2D(64, kernel_size=5, strides=(2, 3), activation='relu', data_format='channels_last', padding='same')(conv_layer)
        conv_layer = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv_layer)
        conv_layer = Conv2D(128, kernel_size=7, strides=(2, 2), activation='relu', data_format='channels_last', padding='same')(conv_layer)
        conv_layer = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv_layer)
        conv_layer = Conv2D(256, kernel_size=7, strides=(2, 2), activation='relu', data_format='channels_last', padding='same')(conv_layer)
        conv_layer = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv_layer)
        dense_layer = Flatten()(conv_layer)
        dense_layer = Dense(512, kernel_regularizer=regularizers.l2(0.001), activation='relu')(dense_layer)
        dense_layer = Dense(H.ls_dim, kernel_regularizer=regularizers.l2(0.001), activation='relu')(dense_layer)
        encoder = Model(inp, dense_layer)
        logger.info("Encoder created")

        dense_layer = Dense(512, kernel_regularizer=regularizers.l2(0.001), activation='relu')(dense_layer)
        dense_layer = Dense(4096, kernel_regularizer=regularizers.l2(0.001), activation='relu')(dense_layer)
        conv_layer = Reshape((4, 4, 256))(dense_layer)
        conv_layer = UpSampling2D(size=(2, 2))(conv_layer)
        conv_layer = Conv2DTranspose(128, kernel_size=8, strides=(1, 1), activation='relu', padding='valid')(conv_layer)
        conv_layer = UpSampling2D(size=(2, 2))(conv_layer)
        conv_layer = Conv2DTranspose(64, kernel_size=7, strides=(2, 2), activation='relu', padding='same')(conv_layer)
        conv_layer = UpSampling2D(size=(2, 2))(conv_layer)
        conv_layer = Conv2DTranspose(32, kernel_size=5, strides=(2, 3), activation='relu', padding='same')(conv_layer)
        conv_layer = Conv2DTranspose(1, kernel_size=3, strides=(1, 1), activation='tanh', padding='same')(conv_layer)
        autoencoder = Model(inp, conv_layer)
        logger.info("Autoencoder created")

        return encoder, autoencoder

    def get_model_lstm(self):
        model = Sequential()
        model.add(LSTM(H.ls_dim, return_sequences=True,
                    input_shape=(H.num_frames, H.ls_dim)))
        model.add(LSTM(H.ls_dim))
        model.add(Dense(16))
        # model.add(RBFLayer(1, 0.5))
        model.add(Dense(1, activation='sigmoid'))
        return model