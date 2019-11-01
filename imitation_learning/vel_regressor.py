from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

import os
import sys
import airsimdroneracingvae

# imports
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..')
sys.path.insert(0, import_path)
import racing_models
import racing_utils


class VelRegressor():
    def __init__(self, regressor_type, bc_weights_path, feature_weights_path=None, latent_space_constraints=True):
        self.regressor_type = regressor_type

        # create models
        if self.regressor_type == 'full':
            self.bc_model = racing_models.bc_full.BcFull()
            self.bc_model.load_weights(bc_weights_path)
        elif self.regressor_type == 'latent':
            # create model
            if latent_space_constraints is True:
                self.cmvae_model = racing_models.cmvae.CmvaeDirect(n_z=10, gate_dim=4, res=64, trainable_model=False)
            else:
                self.cmvae_model = racing_models.cmvae.Cmvae(n_z=10, gate_dim=4, res=64, trainable_model=False)
            self.cmvae_model.load_weights(feature_weights_path)
            self.bc_model = racing_models.bc_latent.BcLatent()
            self.bc_model.load_weights(bc_weights_path)
        elif self.regressor_type == 'reg':
            self.reg_model = racing_models.dronet.Dronet(num_outputs=4, include_top=True)
            self.reg_model.load_weights(feature_weights_path)
            self.bc_model = racing_models.bc_latent.BcLatent()
            self.bc_model.load_weights(bc_weights_path)

    def predict_velocities(self, img, p_o_b):
        img = (img / 255.0) * 2 - 1.0
        if self.regressor_type == 'full':
            predictions = self.bc_model(img)
        elif self.regressor_type == 'latent':
            z, _, _ = self.cmvae_model.encode(img)
            predictions = self.bc_model(z)
        elif self.regressor_type == 'reg':
            z = self.reg_model(img)
            predictions = self.bc_model(z)
        predictions = predictions.numpy()
        predictions = racing_utils.dataset_utils.de_normalize_v(predictions)
        # print('Predicted body vel: \n {}'.format(predictions[0]))
        v_xyz_world = racing_utils.geom_utils.convert_t_body_2_world(airsimdroneracingvae.Vector3r(predictions[0,0], predictions[0,1], predictions[0,2]), p_o_b.orientation)
        return np.array([v_xyz_world.x_val, v_xyz_world.y_val, v_xyz_world.z_val, predictions[0,3]])

