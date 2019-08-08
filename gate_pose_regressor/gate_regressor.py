from __future__ import absolute_import, division, print_function, unicode_literals
import utils_reg

import os
import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))

# import model
models_path = os.path.join(curr_dir, '..', 'racing_models')
sys.path.insert(0, models_path)
import racing_models.dronet

# import utils
models_path = os.path.join(curr_dir, '..', 'racing_utils')
sys.path.insert(0, models_path)
import racing_utils


class GateRegressor():
    def __init__(self, path_weights):
        # set tensorflow variables
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        # 0 = all messages are logged (default behavior)
        # 1 = INFO messages are not printed
        # 2 = INFO and WARNING messages are not printed
        # 3 = INFO, WARNING, and ERROR messages are not printed
        # allow growth is possible using an env var in tf2.0
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        # create model and load weights
        self.model = racing_models.dronet.Dronet(num_outputs=4, include_top=True)
        self.model.load_weights(path_weights)

    def predict_gate_pose(self, img, p_o_b):
        relative_gate_prediction = self.predict_relative_gate_pose(img)
        print('Relative pose: \n {}'.format(relative_gate_prediction[0]))
        p_o_g = racing_utils.geom_utils.getGatePoseWorld(p_o_b, relative_gate_prediction[0,0], relative_gate_prediction[0,1], relative_gate_prediction[0,2], relative_gate_prediction[0,3])
        return p_o_g

    def predict_relative_gate_pose(self, img):
        img = (img / 255.0) * 2 - 1.0
        predictions = self.model(img)
        predictions = predictions.numpy()
        predictions = racing_utils.dataset_utils.de_normalize_gate(predictions)
        return predictions
