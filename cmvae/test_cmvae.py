from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import h5py
import argparse
import tensorflow as tf
from cross_vae_model import CrossVAEModel
from PIL import Image
import matplotlib.pyplot as plt
import pathlib
import glob
from cmvae_utils import create_dataset_csv
from cmvae2_model import Cmvae

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-path', help='model file path', default='/home/rb/catkin_ws/src/AutonomousDrivingCookbook/AirSimE2EDeepLearning/output_64_test5/cmvae_model_100.ckpt', type=str)
# parser.add_argument('--path', '-path', help='model file path', default='/home/rb/catkin_ws/src/AutonomousDrivingCookbook/AirSimE2EDeepLearning/output_128_test/regmodel0005.ckpt', type=str)
parser.add_argument('--n_z', '-n_z', help='size of the each one of the parameters [mean,stddev] in the latent space', default=64, type=int)
# parser.add_argument('--data_dir', '-data_dir', help='path to raw data folder', default='/home/rb/data/airsim_datasets/soccer_relative_poses_polar', type=str)
# parser.add_argument('--data_dir', '-data_dir', help='path to raw data folder', default='/home/rb/data/airsim_datasets/soccer_polar_red_verylarge', type=str)
parser.add_argument('--data_dir', '-data_dir', help='path to raw data folder', default='/home/rb/data/airsim_datasets/soccer_bright_1k', type=str)
parser.add_argument('--n_gates', '-n_gates', help='number of gates that we are encoding from the image', default=4, type=int)
parser.add_argument('--res', '-res', help='destination resolution for images in the cooked data. if 0, do nothing', default=64, type=int)
parser.add_argument('--num_channels', '-num_channels', help='num of channels in image', default=3, type=int)
args = parser.parse_args()


def interp_vector(a, b, n):
    delta = (b-a)/(n-1)
    list_vecs = []
    for i in range(n):
        new_vec = a+delta*i
        list_vecs.append(new_vec)
    return np.asarray(list_vecs)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Create model
model = Cmvae(args.n_z, args.n_gates, trainable_base_model=True, res=args.res)
model.load_weights(args.path)

# Load desired images
files_list = glob.glob(os.path.join(args.data_dir, 'images_test/*.png'))
files_list.sort() # make sure we're reading the images in order later
images_list = []
for file in files_list:
    if args.num_channels == 1:
        im = Image.open(file).resize((args.res, args.res), Image.BILINEAR).convert('L')
        im = np.expand_dims(np.array(im), axis=-1) / 255.0 * 2 - 1.0  # add one more axis and convert to the -1 -> 1 scale
    elif args.num_channels == 3:
        im = Image.open(file).resize((args.res, args.res), Image.BILINEAR)
        im = np.array(im)/255.0*2 - 1.0  # convert to the -1 -> 1 scale
    images_list.append(im)
images_np = np.array(images_list).astype(np.float32)

img_recon, predictions, means, stddev, z = model(images_np, mode=0)
model.summary()

# just for playing with losses:
# flat_pred = tf.reshape(img_recon, [-1])
# flat_gt = tf.reshape(images_np, [-1])
# weights = tf.convert_to_tensor(np.ones(flat_gt.shape), dtype=np.float32)
# error_sq = tf.math.squared_difference(flat_gt, flat_pred)
# softmax_weights = tf.math.exp(error_sq)/tf.reduce_sum(tf.math.exp(error_sq))
# weighted_error_sq = error_sq*softmax_weights
# loss = tf.reduce_sum(weighted_error_sq)
# loss = tf.losses.mean_squared_error()

# de-normalization of distances
r_range = [3.0, 10.0]
cam_fov = 90  # in degrees -- needs to be a bit smaller than 90 in fact because of cone vs. square
alpha = cam_fov / 180.0 * np.pi / 2.0  # alpha is half of fov angle
theta_range = [-alpha, alpha]
psi_range = [np.pi / 2 - alpha, np.pi / 2 + alpha]
eps = np.pi / 15.0
phi_rel_range = [-np.pi + eps, 0 - eps]

predictions = predictions.numpy()

predictions[:, 0] = (predictions[:, 0] + 1.0)/2.0*(r_range[1]-r_range[0]) + r_range[0]
predictions[:, 1] = (predictions[:, 1] + 1.0)/2.0*(theta_range[1]-theta_range[0]) + theta_range[0]
predictions[:, 2] = (predictions[:, 2] + 1.0)/2.0*(psi_range[1]-psi_range[0]) + psi_range[0]
predictions[:, 3] = (predictions[:, 3] + 1.0)/2.0*(phi_rel_range[1]-phi_rel_range[0]) + phi_rel_range[0]

# print('Distance prediction = {}'.format(predictions))

raw_table = np.loadtxt('/home/rb/data/airsim_datasets/soccer_relative_poses_polar/gate_training_data.csv', delimiter=' ')
raw_table.astype(np.float32)

# join predictions with array and print
num_results = 50
indices = np.array([np.arange(num_results)]).transpose()
results = np.concatenate((indices, raw_table[:num_results, :], predictions), axis=1)
# print('Ground truth distance = {}'.format(distances[4]))
print('Img index | Ground-truth values | Predictions: = \n{}'.format(results))

print('Create second model')
# Create model
# model2 = Cmvae(args.n_z, args.n_gates, trainable_base_model=True, res=args.res)
# model2.load_weights(args.path)
predictions, gate_recon, means, stddev, z = model(images_np, mode=0)
predictions = predictions.numpy()

fig = plt.figure(figsize=(20, 20))
columns = 10
rows = 10
for i in range(1, num_results+1):
    idx_orig = (i-1)*2+1
    img_orig = images_np[i - 1, :]
    img_orig = (img_orig + 1) / 2.0 * 255.0
    img_orig = img_orig.astype(int)
    fig.add_subplot(rows, columns, idx_orig)
    if args.num_channels == 1:
        img_orig = np.squeeze(img_orig, -1)
    plt.imshow(img_orig)
    img_rec = predictions[i-1, :]
    img_rec = (img_rec+1)/2.0*255.0
    img_rec = img_rec.astype(int)
    fig.add_subplot(rows, columns, idx_orig+1)
    if args.num_channels == 1:
        img_rec = np.squeeze(img_rec, -1)
    plt.imshow(img_rec)
fig.savefig(os.path.join('/home/rb/Pictures', 'reconstruction_results.png'))
plt.show()
# for i in range(1, num_results+1):
#     idx_orig = (i-1)*2+1
#     img_orig = images_np[i - 1, :]
#     img_orig = (img_orig + 1) / 2.0 * 255.0
#     img_orig = img_orig.astype(int)
#     fig.add_subplot(rows, columns, idx_orig)
#     plt.imshow(img_orig)
#     img_rec = predictions[i-1, :]
#     img_rec = (img_rec+1)/2.0*255.0
#     img_rec = img_rec.astype(int)
#     fig.add_subplot(rows, columns, idx_orig+1)
#     plt.imshow(img_rec)
# fig.savefig(os.path.join('/home/rb/Pictures', 'reconstruction_results.png'))
# plt.show()

# interpolate btw two given images and see the resulting images and distances
idx_close = 7
idx_far = 39
img_close = images_np[idx_close,]
img_far = images_np[idx_far,]
imgs_in = np.array([img_close,img_far])
predictions, _, _ = model.encode(imgs_in, mode=0)
z_close = predictions.numpy()[0,:]
z_far = predictions.numpy()[1,:]
num_interp = 10
z_interp = interp_vector(z_close,z_far, num_interp)

# get the image predictions
img_predictions, gate_predictions = model.decode(z_interp, mode=0)
img_predictions = img_predictions.numpy()

# get the distance predictions
predictions = gate_predictions
predictions = predictions.numpy()
predictions[:, 0] = (predictions[:, 0] + 1.0)/2.0*(r_range[1]-r_range[0]) + r_range[0]
predictions[:, 1] = (predictions[:, 1] + 1.0)/2.0*(theta_range[1]-theta_range[0]) + theta_range[0]
predictions[:, 2] = (predictions[:, 2] + 1.0)/2.0*(psi_range[1]-psi_range[0]) + psi_range[0]
predictions[:, 3] = (predictions[:, 3] + 1.0)/2.0*(phi_rel_range[1]-phi_rel_range[0]) + phi_rel_range[0]

# join predictions with array and print
indices = np.array([np.arange(num_interp)]).transpose()
results = np.concatenate((indices, predictions), axis=1)
# print('Ground truth distance = {}'.format(distances[4]))
print('Img index | Predictions: = \n{}'.format(results))

fig2=plt.figure(figsize=(96, 96))
columns = num_interp+2
rows = 1

img = (img_close + 1) / 2.0 * 255.0
img = img.astype(int)
fig2.add_subplot(rows, columns, 1)
plt.imshow(img)
for i in range(1, num_interp+1):
    img = img_predictions[i - 1,]
    img = (img + 1) / 2.0 * 255.0
    img = img.astype(int)
    fig2.add_subplot(rows, columns, i+1)
    if args.num_channels == 1:
        img = np.squeeze(img, -1)
    plt.imshow(img)
img = (img_far + 1) / 2.0 * 255.0
img = img.astype(int)
fig2.add_subplot(rows, columns, num_interp+2)
plt.imshow(img)
fig2.savefig(os.path.join('/home/rb/Pictures', 'reconstruction_interpolation_results.png'))
plt.show()
