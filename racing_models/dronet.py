import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Lambda, Concatenate, Conv2DTranspose, Reshape, ReLU

class Dronet(Model):
    def __init__(self, num_outputs, include_top=True):
        super(Dronet, self).__init__()
        self.include_top = include_top
        self.create_model(num_outputs)

    def call(self, img):
        # Input
        x1 = self.conv0(img)
        x1 = self.max0(x1)

        # First residual block
        x2 = self.bn0(x1)
        # x2 = x1
        x2 = tf.keras.layers.Activation('relu')(x2)
        x2 = self.conv1(x2)

        x2 = self.bn1(x2)
        x2 = tf.keras.layers.Activation('relu')(x2)
        x2 = self.conv2(x2)

        x1 = self.conv3(x1)
        x3 = tf.keras.layers.add([x1, x2])

        # Second residual block
        x4 = self.bn2(x3)
        # x4 = x3
        x4 = tf.keras.layers.Activation('relu')(x4)
        x4 = self.conv4(x4)

        x4 = self.bn3(x4)
        x4 = tf.keras.layers.Activation('relu')(x4)
        x4 = self.conv5(x4)

        x3 = self.conv6(x3)
        x5 = tf.keras.layers.add([x3, x4])

        # Third residual block
        x6 = self.bn4(x5)
        # x6 = x5
        x6 = tf.keras.layers.Activation('relu')(x6)
        x6 = self.conv7(x6)

        x6 = self.bn5(x6)
        x6 = tf.keras.layers.Activation('relu')(x6)
        x6 = self.conv8(x6)

        x5 = self.conv9(x5)
        x7 = tf.keras.layers.add([x5, x6])

        x = tf.keras.layers.Flatten()(x7)

        if self.include_top:
            x = tf.keras.layers.Activation('relu')(x)
            # x = tf.keras.layers.Dropout(0.5)(x)
            x = self.dense0(x)
            x = self.dense1(x)
            gate_pose = self.dense2(x)
            # phi_rel = self.dense_phi_rel(x)
            # gate_pose = tf.concat([gate_pose, phi_rel], 1)
            return gate_pose
        else:
            return x

    def create_model(self, num_outputs):
        print('[Dronet] Starting dronet')

        self.max0 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)  # default pool_size='2', strides=2

        self.bn0 = tf.keras.layers.BatchNormalization()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.bn5 = tf.keras.layers.BatchNormalization()

        self.conv0 = Conv2D(filters=32, kernel_size=5, strides=2, padding='same', activation='linear')
        self.conv1 = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='linear', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.conv2 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='linear', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.conv3 = Conv2D(filters=32, kernel_size=1, strides=2, padding='same', activation='linear')
        self.conv4 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='linear', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.conv5 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='linear', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.conv6 = Conv2D(filters=64, kernel_size=1, strides=2, padding='same', activation='linear')
        self.conv7 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='linear', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.conv8 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='linear', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.conv9 = Conv2D(filters=128, kernel_size=1, strides=2, padding='same', activation='linear')

        self.dense0 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense1 = tf.keras.layers.Dense(units=32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=num_outputs, activation='linear')
        # self.dense_phi_rel = tf.keras.layers.Dense(units=2, activation='tanh')

        print('[Dronet] Done with dronet')