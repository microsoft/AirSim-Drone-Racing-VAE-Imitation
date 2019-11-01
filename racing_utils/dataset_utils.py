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

def convert_bgr2rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def convert_rgb2bgr(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def normalize_v(v):
    # normalization of velocities from whatever to [-1, 1] range
    v_x_range = [-1, 7]
    v_y_range = [-3, 3]
    v_z_range = [-3, 3]
    v_yaw_range = [-1, 1]
    if len(v.shape) == 1:
        # means that it's a 1D vector of velocities
        v[0] = 2.0 * (v[0] - v_x_range[0]) / (v_x_range[1] - v_x_range[0]) - 1.0
        v[1] = 2.0 * (v[1] - v_y_range[0]) / (v_y_range[1] - v_y_range[0]) - 1.0
        v[2] = 2.0 * (v[2] - v_z_range[0]) / (v_z_range[1] - v_z_range[0]) - 1.0
        v[3] = 2.0 * (v[3] - v_yaw_range[0]) / (v_yaw_range[1] - v_yaw_range[0]) - 1.0
    elif len(v.shape) == 2:
        # means that it's a 2D vector of velocities
        v[:, 0] = 2.0 * (v[:, 0] - v_x_range[0]) / (v_x_range[1] - v_x_range[0]) - 1.0
        v[:, 1] = 2.0 * (v[:, 1] - v_y_range[0]) / (v_y_range[1] - v_y_range[0]) - 1.0
        v[:, 2] = 2.0 * (v[:, 2] - v_z_range[0]) / (v_z_range[1] - v_z_range[0]) - 1.0
        v[:, 3] = 2.0 * (v[:, 3] - v_yaw_range[0]) / (v_yaw_range[1] - v_yaw_range[0]) - 1.0
    else:
        raise Exception('Error in data format of V shape: {}'.format(v.shape))
    return v

def de_normalize_v(v):
    # normalization of velocities from [-1, 1] range to whatever
    v_x_range = [-1, 7]
    v_y_range = [-3, 3]
    v_z_range = [-3, 3]
    v_yaw_range = [-1, 1]
    if len(v.shape) == 1:
        # means that it's a 1D vector of velocities
        v[0] = (v[0] + 1.0) / 2.0 * (v_x_range[1] - v_x_range[0]) + v_x_range[0]
        v[1] = (v[1] + 1.0) / 2.0 * (v_y_range[1] - v_y_range[0]) + v_y_range[0]
        v[2] = (v[2] + 1.0) / 2.0 * (v_z_range[1] - v_z_range[0]) + v_z_range[0]
        v[3] = (v[3] + 1.0) / 2.0 * (v_yaw_range[1] - v_yaw_range[0]) + v_yaw_range[0]
    elif len(v.shape) == 2:
        # means that it's a 2D vector of velocities
        v[:, 0] = (v[:, 0] + 1.0) / 2.0 * (v_x_range[1] - v_x_range[0]) + v_x_range[0]
        v[:, 1] = (v[:, 1] + 1.0) / 2.0 * (v_y_range[1] - v_y_range[0]) + v_y_range[0]
        v[:, 2] = (v[:, 2] + 1.0) / 2.0 * (v_z_range[1] - v_z_range[0]) + v_z_range[0]
        v[:, 3] = (v[:, 3] + 1.0) / 2.0 * (v_yaw_range[1] - v_yaw_range[0]) + v_yaw_range[0]
    else:
        raise Exception('Error in data format of V shape: {}'.format(v.shape))
    return v

def normalize_gate(pose):
    # normalization of velocities from whatever to [-1, 1] range
    r_range = [0.1, 20]
    cam_fov = 90*0.85  # in degrees -- needs to be a bit smaller than 90 in fact because of cone vs. square
    alpha = cam_fov / 180.0 * np.pi / 2.0  # alpha is half of fov angle
    theta_range = [-alpha, alpha]
    psi_range = [np.pi / 2 - alpha, np.pi / 2 + alpha]
    eps = 0.0
    phi_rel_range = [-np.pi + eps, 0 - eps]
    if len(pose.shape) == 1:
        # means that it's a 1D vector of velocities
        pose[0] = 2.0 * (pose[0] - r_range[0]) / (r_range[1] - r_range[0]) - 1.0
        pose[1] = 2.0 * (pose[1] - theta_range[0]) / (theta_range[1] - theta_range[0]) - 1.0
        pose[2] = 2.0 * (pose[2] - psi_range[0]) / (psi_range[1] - psi_range[0]) - 1.0
        pose[3] = 2.0 * (pose[3] - phi_rel_range[0]) / (phi_rel_range[1] - phi_rel_range[0]) - 1.0
    elif len(pose.shape) == 2:
        # means that it's a 2D vector of velocities
        pose[:, 0] = 2.0 * (pose[:, 0] - r_range[0]) / (r_range[1] - r_range[0]) - 1.0
        pose[:, 1] = 2.0 * (pose[:, 1] - theta_range[0]) / (theta_range[1] - theta_range[0]) - 1.0
        pose[:, 2] = 2.0 * (pose[:, 2] - psi_range[0]) / (psi_range[1] - psi_range[0]) - 1.0
        pose[:, 3] = 2.0 * (pose[:, 3] - phi_rel_range[0]) / (phi_rel_range[1] - phi_rel_range[0]) - 1.0
    else:
        raise Exception('Error in data format of V shape: {}'.format(pose.shape))
    return pose

def de_normalize_gate(pose):
    # normalization of velocities from [-1, 1] range to whatever
    r_range = [0.1, 20]
    cam_fov = 90*0.85  # in degrees -- needs to be a bit smaller than 90 in fact because of cone vs. square
    alpha = cam_fov / 180.0 * np.pi / 2.0  # alpha is half of fov angle
    theta_range = [-alpha, alpha]
    psi_range = [np.pi / 2 - alpha, np.pi / 2 + alpha]
    eps = 0.0
    phi_rel_range = [-np.pi + eps, 0 - eps]
    if len(pose.shape) == 1:
        # means that it's a 1D vector of velocities
        pose[0] = (pose[0] + 1.0) / 2.0 * (r_range[1] - r_range[0]) + r_range[0]
        pose[1] = (pose[1] + 1.0) / 2.0 * (theta_range[1] - theta_range[0]) + theta_range[0]
        pose[2] = (pose[2] + 1.0) / 2.0 * (psi_range[1] - psi_range[0]) + psi_range[0]
        pose[3] = (pose[3] + 1.0) / 2.0 * (phi_rel_range[1] - phi_rel_range[0]) + phi_rel_range[0]
    elif len(pose.shape) == 2:
        # means that it's a 2D vector of velocities
        pose[:, 0] = (pose[:, 0] + 1.0) / 2.0 * (r_range[1] - r_range[0]) + r_range[0]
        pose[:, 1] = (pose[:, 1] + 1.0) / 2.0 * (theta_range[1] - theta_range[0]) + theta_range[0]
        pose[:, 2] = (pose[:, 2] + 1.0) / 2.0 * (psi_range[1] - psi_range[0]) + psi_range[0]
        pose[:, 3] = (pose[:, 3] + 1.0) / 2.0 * (phi_rel_range[1] - phi_rel_range[0]) + phi_rel_range[0]
    else:
        raise Exception('Error in data format of V shape: {}'.format(pose.shape))
    return pose

def read_images(data_dir, res, max_size=None):
    print('Going to read image file list')
    files_list = glob.glob(os.path.join(data_dir, 'images/*.png'))
    print('Done. Starting sorting.')
    files_list.sort()  # make sure we're reading the images in order later
    print('Done. Before images_np init')
    if max_size is not None:
        size_data = max_size
    else:
        size_data = len(files_list)
    images_np = np.zeros((size_data, res, res, 3)).astype(np.float32)
    print('Done. Going to read images.')
    idx = 0
    for img_name in files_list:
        # read data in BGR format by default!!!
        # notice that model is going to be trained in BGR
        im = cv2.imread(img_name, cv2.IMREAD_COLOR)
        im = cv2.resize(im, (res, res))
        im = im / 255.0 * 2.0 - 1.0
        images_np[idx, :] = im
        if idx % 10000 == 0:
            print ('image idx = {}'.format(idx))
        idx = idx + 1
        if idx == size_data:
            # reached the last point -- exit loop of images
            break

    print('Done reading {} images.'.format(images_np.shape[0]))
    return images_np

def create_dataset_csv(data_dir, batch_size, res, max_size=None):
    print('Going to read file list')
    files_list = glob.glob(os.path.join(data_dir, 'images/*.png'))
    print('Done. Starting sorting.')
    files_list.sort()  # make sure we're reading the images in order later
    print('Done. Before images_np init')
    if max_size is not None:
        size_data = max_size
    else:
        size_data = len(files_list)
    images_np = np.zeros((size_data, res, res, 3)).astype(np.float32)

    print('Done. Going to read images.')
    idx = 0
    for file in files_list:
        # read data in BGR format by default!!!
        # notice that model is going to be trained in BGR
        im = cv2.imread(file, cv2.IMREAD_COLOR)
        im = cv2.resize(im, (res, res))
        im = im / 255.0 * 2.0 - 1.0
        images_np[idx, :] = im
        if idx % 10000 == 0:
            print ('image idx = {}'.format(idx))
        idx = idx + 1
        if idx == size_data:
            # reached the last point -- exit loop of images
            break

    print('Going to read csv file.')
    # prepare gate R THETA PSI PHI as np array reading from a file
    raw_table = np.loadtxt(data_dir + '/gate_training_data.csv', delimiter=' ')
    raw_table = raw_table[:size_data, :]

    # sanity check
    if raw_table.shape[0] != images_np.shape[0]:
        raise Exception('Number of images ({}) different than number of entries in table ({}): '.format(images_np.shape[0], raw_table.shape[0]))
    raw_table.astype(np.float32)

    # print some useful statistics
    print("Average gate values: {}".format(np.mean(raw_table, axis=0)))
    print("Median  gate values: {}".format(np.median(raw_table, axis=0)))
    print("STD of  gate values: {}".format(np.std(raw_table, axis=0)))
    print("Max of  gate values: {}".format(np.max(raw_table, axis=0)))
    print("Min of  gate values: {}".format(np.min(raw_table, axis=0)))

    # normalize distances to gate to [-1, 1] range
    raw_table = normalize_gate(raw_table)

    img_train, img_test, dist_train, dist_test = train_test_split(images_np, raw_table, test_size=0.1, random_state=42)

    # convert to tf format dataset and prepare batches
    ds_train = tf.data.Dataset.from_tensor_slices((img_train, dist_train)).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices((img_test, dist_test)).batch(batch_size)

    return ds_train, ds_test

def create_unsup_dataset_multiple_sources(data_dir_list, batch_size, res):
    # load all the images in one single large dataset
    images_np = np.empty((0,res,res,3)).astype(np.float32)
    for data_dir in data_dir_list:
        img_array = read_images(data_dir, res, max_size=None)
        images_np = np.concatenate((images_np, img_array), axis=0)
    # make fake distances to gate as -1
    num_items = images_np.shape[0]
    print('Real_life dataset has {} images total'.format(num_items))
    raw_table = (-1.0*np.ones((num_items, 4))).astype(np.float32)
    # separate the actual dataset:
    img_train, img_test, dist_train, dist_test = train_test_split(images_np, raw_table, test_size=0.1, random_state=42)
    # convert to tf format dataset and prepare batches
    ds_train = tf.data.Dataset.from_tensor_slices((img_train, dist_train)).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices((img_test, dist_test)).batch(batch_size)
    return ds_train, ds_test

def create_test_dataset_csv(data_dir, res, read_table=True):
    # prepare image dataset from a folder
    print('Going to read file list')
    files_list = glob.glob(os.path.join(data_dir, 'images/*.png'))
    print('Done. Starting sorting.')
    files_list.sort()  # make sure we're reading the images in order later
    print('Done. Before images_np init')
    images_np = np.zeros((len(files_list), res, res, 3)).astype(np.float32)
    print('After images_np init')
    idx = 0
    for file in files_list:
        # read data in BGR format by default!!!
        # notice that model was trained in BGR
        im = cv2.imread(file, cv2.IMREAD_COLOR)
        im = cv2.resize(im, (res, res))
        im = im/255.0*2.0-1.0
        images_np[idx, :] = im
        idx = idx + 1

    if not read_table:
        return images_np, None

    # prepare gate R THETA PSI PHI as np array reading from a file
    raw_table = np.loadtxt(data_dir + '/gate_training_data.csv', delimiter=' ')
    # sanity check
    if raw_table.shape[0] != images_np.shape[0]:
        raise Exception('Number of images ({}) different than number of entries in table ({}): '.format(images_np.shape[0], raw_table.shape[0]))
    raw_table.astype(np.float32)

    # print some useful statistics
    print("Average gate values: {}".format(np.mean(raw_table, axis=0)))
    print("Median  gate values: {}".format(np.median(raw_table, axis=0)))
    print("STD of  gate values: {}".format(np.std(raw_table, axis=0)))
    print("Max of  gate values: {}".format(np.max(raw_table, axis=0)))
    print("Min of  gate values: {}".format(np.min(raw_table, axis=0)))

    return images_np, raw_table

def create_dataset_txt(data_dir, batch_size, res, data_mode='train', base_path=None):
    vel_table = np.loadtxt(data_dir + '/proc_vel.txt', delimiter=',').astype(np.float32)
    with open(data_dir + '/proc_images.txt') as f:
        img_table = f.read().splitlines()

    # sanity check
    if vel_table.shape[0] != len(img_table):
        raise Exception('Number of images ({}) different than number of entries in table ({}): '.format(len(img_table), vel_table.shape[0]))

    size_data = len(img_table)
    images_np = np.zeros((size_data, res, res, 3)).astype(np.float32)

    print('Done. Going to read images.')
    idx = 0
    for img_name in img_table:
        if base_path is not None:
            img_name = img_name.replace('/home/rb/data', base_path)
        # read data in BGR format by default!!!
        # notice that model is going to be trained in BGR
        im = cv2.imread(img_name, cv2.IMREAD_COLOR)
        im = cv2.resize(im, (res, res))
        im = im / 255.0 * 2.0 - 1.0
        images_np[idx, :] = im
        if idx % 1000 == 0:
            print ('image idx = {} out of {} images'.format(idx, size_data))
        idx = idx + 1
        if idx == size_data:
            # reached the last point -- exit loop of images
            break

    # print some useful statistics and normalize distances
    print("Num samples: {}".format(vel_table.shape[0]))
    print("Average vx: {}".format(np.mean(vel_table[:, 0])))
    print("Average vy: {}".format(np.mean(vel_table[:, 1])))
    print("Average vz: {}".format(np.mean(vel_table[:, 2])))
    print("Average vyaw: {}".format(np.mean(vel_table[:, 3])))

    # normalize the values of velocities to the [-1, 1] range
    vel_table = normalize_v(vel_table)

    img_train, img_test, v_train, v_test = train_test_split(images_np, vel_table, test_size=0.1, random_state=42)

    if data_mode == 'train':
        # convert to tf format dataset and prepare batches
        ds_train = tf.data.Dataset.from_tensor_slices((img_train, v_train)).batch(batch_size)
        ds_test = tf.data.Dataset.from_tensor_slices((img_test, v_test)).batch(batch_size)
        return ds_train, ds_test
    elif data_mode == 'test':
        return img_test, v_test

def create_dataset_multiple_sources(data_dir_list, batch_size, res, data_mode='train', base_path=None):
    # load all the images and velocities in one single large dataset
    images_np = np.empty((0,res,res,3)).astype(np.float32)
    vel_table = np.empty((0,4)).astype(np.float32)
    for data_dir in data_dir_list:
        img_array, v_array = create_dataset_txt(data_dir, batch_size, res, data_mode='test', base_path=base_path)
        images_np = np.concatenate((images_np, img_array), axis=0)
        vel_table = np.concatenate((vel_table, v_array), axis=0)
    # separate the actual dataset:
    img_train, img_test, v_train, v_test = train_test_split(images_np, vel_table, test_size=0.1, random_state=42)
    if data_mode == 'train':
        # convert to tf format dataset and prepare batches
        ds_train = tf.data.Dataset.from_tensor_slices((img_train, v_train)).batch(batch_size)
        ds_test = tf.data.Dataset.from_tensor_slices((img_test, v_test)).batch(batch_size)
        return ds_train, ds_test
    elif data_mode == 'test':
        return img_test, v_test