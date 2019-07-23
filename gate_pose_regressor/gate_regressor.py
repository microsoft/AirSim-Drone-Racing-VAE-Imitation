from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
from regressor_model import Regressor
import utils


class GateRegressor():
    def __init__(self, res, path_weights):
        # process inputs
        self.res = res

        # constants for conversions
        # de-normalization of distances
        self.r_range = [3.0, 10.0]
        cam_fov = 90  # in degrees -- needs to be a bit smaller than 90 in fact because of cone vs. square
        alpha = cam_fov / 180.0 * np.pi / 2.0  # alpha is half of fov angle
        self.theta_range = [-alpha, alpha]
        self.psi_range = [np.pi / 2 - alpha, np.pi / 2 + alpha]
        eps = np.pi / 15.0
        self.phi_rel_range = [-np.pi + eps, 0 - eps]

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

    def predict_gate_pose(self, img, p_o_b):
        relative_gate_prediction = self.predict_relative_gate_pose(img)
        p_o_g = self.calc_global_gate_pose(p_o_b, relative_gate_prediction)
        return p_o_g

    def predict_relative_gate_pose(self, img):
        img = (img / 255.0) * 2 - 1.0
        predictions = self.model(img)
        return self.net_output_2_pose(predictions)

    def net_output_2_pose(self, predictions):
        predictions = predictions.numpy()
        predictions[:, 0] = (predictions[:, 0] + 1.0) / 2.0 * (self.r_range[1] - self.r_range[0]) + self.r_range[0]
        predictions[:, 1] = (predictions[:, 1] + 1.0) / 2.0 * (self.theta_range[1] - self.theta_range[0]) + self.theta_range[0]
        predictions[:, 2] = (predictions[:, 2] + 1.0) / 2.0 * (self.psi_range[1] - self.psi_range[0]) + self.psi_range[0]
        predictions[:, 3] = (predictions[:, 3] + 1.0) / 2.0 * (self.phi_rel_range[1] - self.phi_rel_range[0]) + self.phi_rel_range[0]
        return predictions

    def calc_global_gate_pose(self, p_o_b, relative_pose):
        r = relative_pose[0]
        theta = relative_pose[1]
        psi = relative_pose[2]
        phi_rel = relative_pose[3]
        # get relative vector in the base frame
        t_b_g = utils.polarTranslation(r, theta, psi)
        p_o_g = utils.convert_gate_base2world(p_o_b, t_b_g, phi_rel)
        return p_o_g
