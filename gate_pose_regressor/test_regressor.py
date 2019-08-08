from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import argparse
import tensorflow as tf
from PIL import Image
import glob
import matplotlib.pyplot as plt
import cv2

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

###########################################

# DEFINE TESTING META PARAMETERS
data_dir = '/home/rb/data/airsim_datasets/soccer_bright_1k'
weights_path = '/home/rb/data/model_outputs/reg_5/reg_model_20.ckpt'
img_res = 96

###########################################

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Create model
model = racing_models.dronet.Dronet(num_outputs=4, include_top=True)
model.load_weights(weights_path)

# Load test dataset
images_np, raw_table = racing_utils.dataset_utils.create_test_dataset_csv(data_dir, img_res, num_channels=3)

predictions = model(images_np)
predictions = predictions.numpy()

# de-normalization of distances
predictions = racing_utils.dataset_utils.de_normalize_gate(predictions)
gt = racing_utils.dataset_utils.de_normalize_gate(raw_table)

# show images in a loop
# for idx in range(images_np.shape[1]):
#     cv2.imshow('image', images_np[idx, :])
#     cv2.waitKey(1000)

# calculate statistics with respect to ground-truth values
racing_utils.stats_utils.calculate_gate_stats(predictions, gt)

