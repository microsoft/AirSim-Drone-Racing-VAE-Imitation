from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
from bc_model import Regressor
import utils_bc


class GateRegressor():
    def __init__(self, res, path_weights):
        # process inputs
        self.res = res

        # constants for conversions
        # de-normalization of distances
        self.v_max_lin = [0.0, 5.0]
        self.v_max_ang = [-0.5, 0]

        # set tensorflow variables
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        # 0 = all messages are logged (default behavior)
        # 1 = INFO messages are not printed
        # 2 = INFO and WARNING messages are not printed
        # 3 = INFO, WARNING, and ERROR messages are not printed
        # allow growth is possible using an env var in tf2.0
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        # create model and load weights
        self.res = 96
        self.model = Regressor(trainable_base_model=True, res=self.res)
        self.model.load_weights(path_weights)

    def predict_velocities(self, img, p_o_b):
        vel_prediction_local = self.predict_vel(img)
        vel_prediction_global = self.calc_global_vel(vel_prediction_local, p_o_b)
        return vel_prediction_global

    def predict_vel(self, img):
        img = (img / 255.0) * 2 - 1.0
        predictions = self.model(img)
        return self.net_output_2_vel(predictions)

    def net_output_2_vel(self, predictions):
        predictions = predictions.numpy()
        predictions[:, 0] = (predictions[:, 0] + 1.0) / 2.0 * (self.v_max_lin[1] - self.v_max_lin[0]) + self.v_max_lin[0]
        predictions[:, 1] = (predictions[:, 1] + 1.0) / 2.0 * (self.v_max_lin[1] - self.v_max_lin[0]) + self.v_max_lin[0]
        predictions[:, 2] = (predictions[:, 2] + 1.0) / 2.0 * (self.v_max_lin[1] - self.v_max_lin[0]) + self.v_max_lin[0]
        predictions[:, 3] = (predictions[:, 3] + 1.0) / 2.0 * (self.v_max_ang[1] - self.v_max_ang[0]) + self.v_max_ang[0]
        return predictions

    def calc_global_vel(self, vel_prediction_local, p_o_b):
        vx = vel_prediction_local[0, 0]
        vy = vel_prediction_local[0, 1]
        vz = vel_prediction_local[0, 2]
        v_yaw = vel_prediction_local[0, 3]
        # get velocities in the base frame
        p_o_g = utils_bc.convert_vel_base2world(p_o_b, vx, vy, vz, v_yaw)
        return p_o_g
