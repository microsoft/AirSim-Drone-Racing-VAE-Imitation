import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt
import glob
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split


def create_dataset_csv(data_dir, batch_size, res, num_channels):
    # prepare image dataset from a folder
    files_list = glob.glob(os.path.join(data_dir, 'images/*.png'))
    files_list.sort() # make sure we're reading the images in order later
    images_list = []
    for file in files_list:
        if num_channels == 1:
            im = Image.open(file).resize((res, res), Image.BILINEAR).convert('L')
            im = np.expand_dims(np.array(im), axis=-1) / 255.0 * 2 - 1.0  # add one more axis and convert to the -1 -> 1 scale
        elif num_channels == 3:
            im = Image.open(file).resize((res, res), Image.BILINEAR)
            im = np.array(im)/255.0*2 - 1.0  # convert to the -1 -> 1 scale
        images_list.append(im)
    images_np = np.array(images_list).astype(np.float32)

    # prepare gate R THETA PSI PHI as np array reading from a file
    raw_table = np.loadtxt(data_dir + '/gate_training_data.csv', delimiter=' ')
    # sanity check
    if raw_table.shape[0] != images_np.shape[0]:
        raise Exception('Number of images ({}) different than number of entries in table ({}): '.format(images_np.shape[0], raw_table.shape[0]))

    raw_table.astype(np.float32)

    # normalization of distances
    r_range = [3.0, 10.0]
    cam_fov = 90  # in degrees -- needs to be a bit smaller than 90 in fact because of cone vs. square
    alpha = cam_fov / 180.0 * np.pi / 2.0  # alpha is half of fov angle
    theta_range = [-alpha, alpha]
    psi_range = [np.pi / 2 - alpha, np.pi / 2 + alpha]
    eps = np.pi / 15.0
    phi_rel_range = [-np.pi + eps, 0 - eps]

    # print some useful statistics and normalize distances
    print("Average distance to gate: {}".format(np.mean(raw_table[:,0])))
    print("Median distance to gate: {}".format(np.median(raw_table[:,0])))
    print("STD of distance to gate: {}".format(np.std(raw_table[:,0])))

    raw_table[:, 0] = 2.0 * (raw_table[:, 0] - r_range[0]) / (r_range[1] - r_range[0]) - 1.0
    raw_table[:, 1] = 2.0 * (raw_table[:, 1] - theta_range[0]) / (theta_range[1] - theta_range[0]) - 1.0
    raw_table[:, 2] = 2.0 * (raw_table[:, 2] - psi_range[0]) / (psi_range[1] - psi_range[0]) - 1.0
    raw_table[:, 3] = 2.0 * (raw_table[:, 3] - phi_rel_range[0]) / (phi_rel_range[1] - phi_rel_range[0]) - 1.0

    img_train, img_test, dist_train, dist_test = train_test_split(images_np, raw_table, test_size=0.1, random_state=42)

    # convert to tf format dataset and prepare batches
    ds_train = tf.data.Dataset.from_tensor_slices((img_train, dist_train)).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices((img_test, dist_test)).batch(batch_size)

    return ds_train, ds_test
