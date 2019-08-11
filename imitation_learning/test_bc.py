import tensorflow as tf
import os
import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))

# import model
models_path = os.path.join(curr_dir, '..', 'racing_models')
sys.path.insert(0, models_path)
import racing_models

# import utils
models_path = os.path.join(curr_dir, '..', 'racing_utils')
sys.path.insert(0, models_path)
import racing_utils

###########################################

# DEFINE TRAINING META PARAMETERS
data_dir = '/home/rb/data/il_datasets/il_3'
training_mode = 'latent'  # 'full' or 'latent'
bc_weights_path = '/home/rb/data/model_outputs/bc_latent_2/bc_model_270.ckpt'
cmvae_weights_path = '/home/rb/data/model_outputs/cmvae_9/cmvae_model_20.ckpt'
n_z = 20
batch_size = 64
epochs = 10000
img_res = 64
max_size = None  # default is None

###########################################

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# load test dataset
print('Starting dataset')
images_np, v_gt = racing_utils.dataset_utils.create_dataset_txt(data_dir, batch_size, img_res, data_mode='test')
print('Done with dataset')

# create models
if training_mode == 'full':
    bc_model = racing_models.bc_full.BcFull()
    bc_model.load_weights(bc_weights_path)
    predictions = bc_model(images_np)
elif training_mode == 'latent':
    cmvae_model = racing_models.cmvae.Cmvae(n_z=n_z, gate_dim=4, res=img_res)
    cmvae_model.load_weights(cmvae_weights_path)
    bc_model = racing_models.bc_latent.BcLatent()
    bc_model.load_weights(bc_weights_path)
    z, _, _ = cmvae_model.encode(images_np)
    predictions = bc_model(z)

predictions = predictions.numpy()

# de-normalization of velocities
predictions = racing_utils.dataset_utils.de_normalize_v(predictions)
v_gt = racing_utils.dataset_utils.de_normalize_v(v_gt)

# calculate statistics with respect to ground-truth values
racing_utils.stats_utils.calculate_v_stats(predictions, v_gt)

