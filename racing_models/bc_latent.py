import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Lambda, Concatenate, Conv2DTranspose, Reshape, ReLU

class BcLatent(Model):
    def __init__(self):
        super(BcLatent, self).__init__()
        self.create_model()

    def call(self, z):
        return self.network(z)

    def create_model(self):
        print('[BcLatent] Starting create_model')
        dense0 = tf.keras.layers.Dense(units=256, activation='relu')
        dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        dense2 = tf.keras.layers.Dense(units=64, activation='relu')
        dense3 = tf.keras.layers.Dense(units=16, activation='relu')
        dense4 = tf.keras.layers.Dense(units=4, activation='linear')
        self.network = tf.keras.Sequential([
            dense0,
            # dense1,
            # dense2,
            dense3,
            dense4], 
            name='bc_dense')
        print('[BcLatent] Done with create_model')