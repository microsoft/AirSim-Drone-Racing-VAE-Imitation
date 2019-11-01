import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Lambda, Concatenate, Conv2DTranspose, Reshape, ReLU

class NonLinearTransformer(Model):
    def __init__(self):
        super(NonLinearTransformer, self).__init__()
        self.create_model()

    def call(self, x):
        return self.network(x)

    def create_model(self):
        print('[NonLinearTransformer] Starting create_model')
        dense0 = tf.keras.layers.Dense(units=64, activation='relu')
        dense1 = tf.keras.layers.Dense(units=32, activation='relu')
        dense2 = tf.keras.layers.Dense(units=1, activation='linear')
        self.network = tf.keras.Sequential([
            dense0,
            dense1,
            dense2], 
            name='nonlineartransformer')

        print('[NonLinearTransformer] Done with create_model')

class TestNet(Model):
    def __init__(self):
        super(TestNet, self).__init__()
        self.create_model()

    def call(self, x):
        x = tf.keras.layers.Flatten()(x)
        return self.network(x)

    def create_model(self):
        print('[NonLinearTransformer] Starting create_model')
        dense0 = tf.keras.layers.Dense(units=64, activation='relu')
        dense1 = tf.keras.layers.Dense(units=32, activation='relu')
        dense2 = tf.keras.layers.Dense(units=1, activation='linear')
        self.network = tf.keras.Sequential([
            dense0,
            dense1,
            dense2], 
            name='nonlineartransformer')

        print('[NonLinearTransformer] Done with create_model')