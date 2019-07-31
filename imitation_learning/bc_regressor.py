from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
from bc_model import ImgRegressor
import bc_utils


class VelRegressor():
    def __init__(self, res, path_weights):
        # process inputs
        self.res = res

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
        self.model = ImgRegressor(trainable_base_model=True, res=self.res)
        self.model.load_weights(path_weights)

    def predict_velocities(self, img, p_o_b):
        vel_prediction_local = self.predict_vel(img)
        v_local = np.zeros((4,))
        v_local[0] = vel_prediction_local[0, 0]
        v_local[1] = vel_prediction_local[0, 1]
        v_local[2] = vel_prediction_local[0, 2]
        v_local[3] = vel_prediction_local[0, 3]
        v_global = bc_utils.v_body_to_world(v_local, p_o_b)
        return v_global

    def predict_vel(self, img):
        img = (img / 255.0) * 2 - 1.0
        predictions = self.model(img)
        predictions = predictions.numpy()
        # print(predictions)
        predictions = bc_utils.de_normalize_v(predictions)
        return predictions

