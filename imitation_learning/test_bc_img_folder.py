import tensorflow as tf
import os
import sys
import cv2
import numpy as np
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
save_video = True
video_file = '/home/rb/Videos/real_life/bc_con_slow_video_7.avi'
data_dir = '/home/rb/data/real_life/video_7'
training_mode = 'latent'  # 'full' or 'latent'
bc_weights_path = '/home/rb/data/model_outputs/bc_con_slow/bc_model_80.ckpt'
cmvae_weights_path = '/home/rb/data/model_outputs/cmvae_con/cmvae_model_40.ckpt'
n_z = 10
img_res = 64
img_display_res = 400
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
images_np = racing_utils.dataset_utils.read_images(data_dir, img_res, max_size=None)
print('Done with dataset')

# create models
if training_mode == 'full':
    bc_model = racing_models.bc_full.BcFull()
    bc_model.load_weights(bc_weights_path)
    predictions = bc_model(images_np)
elif training_mode == 'latent':
    # cmvae_model = racing_models.cmvae.Cmvae(n_z=n_z, gate_dim=4, res=img_res)
    cmvae_model = racing_models.cmvae.CmvaeDirect(n_z=n_z, gate_dim=4, res=img_res)
    cmvae_model.load_weights(cmvae_weights_path)
    bc_model = racing_models.bc_latent.BcLatent()
    bc_model.load_weights(bc_weights_path)
    z, _, _ = cmvae_model.encode(images_np)
    predictions = bc_model(z)

predictions = predictions.numpy()

# de-normalization of velocities
predictions = racing_utils.dataset_utils.de_normalize_v(predictions)

# visualize velocities for each frame and save video if wanted
if save_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_file, fourcc, 20.0, (img_display_res, img_display_res))
images_np = ((images_np+1.0)/2.0*255.0).astype(np.uint8)
vel_scale = 20
yaw_scale = 40
for img_idx in range(predictions.shape[0]):
    img = images_np[img_idx,:]
    img = cv2.resize(img, (img_display_res, img_display_res))
    o_x = int(img.shape[0]/2)
    o_y = int(img.shape[1]/2)
    origin = (o_x, o_y)
    pt_vx = (o_x, o_y - int(vel_scale * predictions[img_idx, 0]))
    pt_vy = (o_x + int(vel_scale * predictions[img_idx, 1]), o_y)
    cv2.arrowedLine(img, origin, pt_vx, (210, 0, 255), 3)
    cv2.arrowedLine(img, origin, pt_vy, (0, 255, 0), 3)
    cv2.imshow('image', img)
    if save_video:
        out.write(img)
    # time.sleep(0.5)
    cv2.waitKey(20)

if save_video:
    out.release()
cv2.destroyAllWindows()