import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Lambda, Concatenate, Conv2DTranspose, Reshape


class Regressor(Model):
    def __init__(self, trainable_base_model=True, res=96):
        super(Regressor, self).__init__()

        self.res = res
        self.create_q_img(trainable_base_model)
        self.create_p_vel()

    def call(self, x):
        # Encoding
        x = self.q_img(x)
        x = self.p_vel(x)
        return x

    def create_q_img(self, trainable_base_model):
        print('[Cmvae] Starting q_img')
        im_shape = (self.res, self.res, 3)
        # Pre-trained model with MobileNetV2
        print('[Cmvae] Creating base model')
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=im_shape,
            include_top=False,
            weights='imagenet',
            # weights=None,
            alpha=1.0
        )
        print('[Cmvae] Done creating base model')

        # Freeze the pre-trained model weights
        base_model.trainable = trainable_base_model
        # Refreeze layers until the layers we want to fine-tune
        # for layer in base_model.layers[:100]:
        #   layer.trainable =  False

        # Regression head
        # maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
        maxpool_layer = tf.keras.layers.GlobalAveragePooling2D()
        # flatten = Flatten()
        reg_pre_layer = tf.keras.layers.Dense(units=128, activation='relu')
        # self.reg_layer = tf.keras.layers.Dense(units=2 * self.n_z, activation='linear')
        # Layer create custom model for depth regression
        self.q_img = tf.keras.Sequential([
            base_model,
            maxpool_layer,
            # flatten,
            reg_pre_layer
        ], name='q_img')

        print('[Cmvae] Done with q_img')

    def create_p_vel(self):
        print('[Cmvae] Starting p_gate')
        # d3 = Dense(units=128, activation='relu')
        d4 = Dense(units=64, activation='relu')
        # self.d5 = Dense(units=4*self.n_gates, trainable=trainable_decoder)
        d5 = Dense(units=4, activation='linear')  # 4 numerical outputs for vx, vy, vz, vyaw
        self.p_vel = tf.keras.Sequential([
            # d3,
            d4,
            d5
        ], name='p_gate')
        print('[Cmvae] Done with p_gate')
