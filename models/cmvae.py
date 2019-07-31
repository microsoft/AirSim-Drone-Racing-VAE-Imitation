import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.activations import softplus, relu
from tensorflow.keras.backend import random_normal
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Lambda, Concatenate, Conv2DTranspose, Reshape
import dronet
import decoders


# model definition class
class Cmvae(Model):
    def __init__(self, n_z, gate_dim=4, res=96, trainable_model=True):
        super(Cmvae, self).__init__()
        # create the 3 base models:
        self.q_img = dronet.Dronet(num_outputs=n_z*2, include_top=True)
        self.p_img = decoders.ImgDecoder()
        self.p_gate = decoders.GateDecoder(gate_dim=gate_dim)
        # Create sampler
        self.mean_params = Lambda(lambda x: x[:, : n_z])
        self.stddev_params = Lambda(lambda x: x[:, n_z:])

    def call(self, x, mode):
        # Possible modes for reconstruction:
        # 0: img -> img + gate
        # 1: img -> img
        # 2: img -> gate
        x = self.q_img(x)
        means = self.mean_params(x)
        stddev = tf.math.exp(0.5 * self.stddev_params(x))
        eps = random_normal(tf.shape(stddev))
        z = means + eps * stddev
        if mode == 0:
            img_recon = self.p_img(z)
            gate_recon = self.p_gate(z)
            return img_recon, gate_recon, means, stddev, z
        elif mode == 1:
            img_recon = self.p_img(z)
            gate_recon = False
            return img_recon, gate_recon, means, stddev, z
        elif mode == 2:
            img_recon = False
            gate_recon = self.p_gate(z)
            return img_recon, gate_recon, means, stddev, z

    def encode(self, x, mode):
        x = self.q_img(x)
        means = self.mean_params(x)
        stddev = tf.math.exp(0.5 * self.stddev_params(x))
        eps = random_normal(tf.shape(stddev))
        z = means + eps * stddev
        return z, means, stddev

    def decode(self, z, mode):
        # Possible modes for reconstruction:
        # 0: z -> img + gate
        # 1: z -> img
        # 2: z -> gate
        if mode == 0:
            img_recon = self.p_img(z)
            gate_recon = self.p_gate(z)
            return img_recon, gate_recon
        elif mode == 1:
            img_recon = self.p_img(z)
            gate_recon = False
            return img_recon, gate_recon
        elif mode == 2:
            img_recon = False
            gate_recon = self.p_gate(z)
            return img_recon, gate_recon
