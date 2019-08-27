import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
curr_dir = os.path.dirname(os.path.abspath(__file__))

# import model
models_path = os.path.join(curr_dir, '..', 'racing_models')
sys.path.insert(0, models_path)
import racing_models.cmvae

# import utils
models_path = os.path.join(curr_dir, '..', 'racing_utils')
sys.path.insert(0, models_path)
import racing_utils

###########################################

# DEFINE TESTING META PARAMETERS
data_dir = '/home/rb/data/airsim_datasets/soccer_new_1k'
# data_dir = '/home/rb/data/real_life/video_0'
# data_dir = '/home/rb/data/real_life/hand_picked_0'
weights_path = '/home/rb/data/model_outputs/cmvae_test/cmvae_model_85.ckpt'
n_z = 20
img_res = 64
read_table = False

num_imgs_display = 50
columns = 10
rows = 10

num_interp_z = 10
idx_close = 7
idx_far = 39

z_range_mural = [-1.0, 1.0]
z_num_mural = 10

###########################################

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Load test dataset
images_np, raw_table = racing_utils.dataset_utils.create_test_dataset_csv(data_dir, img_res, read_table=read_table)
print('Done with dataset')

# create model
model = racing_models.cmvae.Cmvae(n_z=n_z, gate_dim=4, res=img_res, trainable_model=True)
model.load_weights(weights_path)

img_recon, gate_recon, means, stddev, z = model(images_np, mode=0)
img_recon = img_recon.numpy()
gate_recon = gate_recon.numpy()
z = z.numpy()

# de-normalization of gates and images
images_np = ((images_np + 1.0) / 2.0 * 255.0).astype(np.uint8)
img_recon = ((img_recon + 1.0) / 2.0 * 255.0).astype(np.uint8)
gate_recon = racing_utils.dataset_utils.de_normalize_gate(gate_recon)

# if not read_table:
#     sys.exit()

# get stats for gate reconstruction
# racing_utils.stats_utils.calculate_gate_stats(gate_recon, raw_table)

# show some reconstruction figures
# fig = plt.figure(figsize=(20, 20))
# for i in range(1, num_imgs_display+1):
#     idx_orig = (i-1)*2+1
#     fig.add_subplot(rows, columns, idx_orig)
#     img_display = racing_utils.dataset_utils.convert_bgr2rgb(images_np[i - 1, :])
#     plt.imshow(img_display)
#     fig.add_subplot(rows, columns, idx_orig+1)
#     img_display = racing_utils.dataset_utils.convert_bgr2rgb(img_recon[i-1, :])
#     plt.imshow(img_display)
# fig.savefig(os.path.join('/home/rb/Pictures', 'reconstruction_results.png'))
# plt.show()
#
# # show interpolation btw two images in latent space
# z_close = z[idx_close, :]
# z_far = z[idx_far, :]
# z_interp = racing_utils.geom_utils.interp_vector(z_close, z_far, num_interp_z)
#
# # get the image predictions
# img_recon_interp, gate_recon_interp = model.decode(z_interp, mode=0)
# img_recon_interp = img_recon_interp.numpy()
# gate_recon_interp = gate_recon_interp.numpy()
#
# # de-normalization of gates and images
# img_recon_interp = ((img_recon_interp + 1.0) / 2.0 * 255.0).astype(np.uint8)
# gate_recon_interp = racing_utils.dataset_utils.de_normalize_gate(gate_recon_interp)
#
# # join predictions with array and print
# indices = np.array([np.arange(num_interp_z)]).transpose()
# results = np.concatenate((indices, gate_recon_interp), axis=1)
# print('Img index | Predictions: = \n{}'.format(results))
#
# # plot the interpolated images
# fig2 = plt.figure(figsize=(96, 96))
# columns = num_interp_z + 2
# rows = 1
# fig2.add_subplot(rows, columns, 1)
# img_display = racing_utils.dataset_utils.convert_bgr2rgb(images_np[idx_close, :])
# plt.imshow(img_display)
# for i in range(1, num_interp_z + 1):
#     fig2.add_subplot(rows, columns, i+1)
#     img_display = racing_utils.dataset_utils.convert_bgr2rgb(img_recon_interp[i - 1, :])
#     plt.imshow(img_display)
# fig2.add_subplot(rows, columns, num_interp_z + 2)
# img_display = racing_utils.dataset_utils.convert_bgr2rgb(images_np[idx_far, :])
# plt.imshow(img_display)
# fig2.savefig(os.path.join('/home/rb/Pictures', 'reconstruction_interpolation_results.png'))
# plt.show()


# new plot traveling through latent space
fig3 = plt.figure(figsize=(96, 96))
columns = num_interp_z
rows = n_z
z_values = racing_utils.geom_utils.interp_vector(z_range_mural[0], z_range_mural[1], z_num_mural)
for i in range(1, num_interp_z + 1):
    fig3.add_subplot(rows, columns, i+1)
    z = np.zeros((n_z,1)).astype(np.float32)
    z[i/columns] = z_values[i%columns-1]
    img_recon_interp, _ = model.decode(num_interp_z, mode=0)
    img_recon_interp = img_recon_interp.numpy()
    img_recon_interp = ((img_recon_interp + 1.0) / 2.0 * 255.0).astype(np.uint8)
    img_display = racing_utils.dataset_utils.convert_bgr2rgb(img_recon_interp)
    plt.imshow(img_display)
fig3.savefig(os.path.join('/home/rb/Pictures', 'z_mural.png'))
plt.show()




















