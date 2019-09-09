from __future__ import division
import time
import numpy as np
import vel_regressor
import cv2
import math
import tensorflow as tf

import os, sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
airsim_path = os.path.join(curr_dir, '..', 'airsim')
sys.path.insert(0, airsim_path)
import setup_path
import airsim
import airsim.types
import airsim.utils

# import utils
models_path = os.path.join(curr_dir, '..', 'racing_utils')
sys.path.insert(0, models_path)
import racing_utils


def process_image(client, img_res):
    image_response = client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
    img_1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)  # get numpy array
    img_bgr = img_1d.reshape(image_response.height, image_response.width, 3)  # reshape array to 4 channel image array H X W X 3
    img_resized = cv2.resize(img_bgr, (img_res, img_res)).astype(np.float32)
    img_batch_1 = np.array([img_resized])
    cam_pos = image_response.camera_position
    cam_orientation = image_response.camera_orientation
    return img_batch_1, cam_pos, cam_orientation


def move_drone(client, vel_cmd):
    # good multipliers originally: 0.4 for vel, 0.8 for yaw
    # good multipliers new policies: 0.8 for vel, 0.8 for yaw
    vel_cmd[0:2] = vel_cmd[0:2] *1.0  # usually base speed is 3/ms
    vel_cmd[3] = vel_cmd[3] * 1.0
    # yaw rate is given in deg/s!! not rad/s
    yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=vel_cmd[3]*180.0/np.pi)
    client.moveByVelocityAsync(vel_cmd[0], vel_cmd[1], vel_cmd[2], duration=0.1, yaw_mode=yaw_mode)


print(os.path.abspath(airsim.__file__))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
# tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')
# tf.debugging.set_log_device_placement(True)


if __name__ == "__main__":
    # set airsim client
    client = airsim.MultirotorClient()
    client.confirmConnection()
    # client.simLoadLevel('Soccer_Field_Easy')
    client.simLoadLevel('Soccer_Field_Easy')
    time.sleep(2)
    # should match the names in settings.json
    drone_name = "drone_0"

    client.enableApiControl(True, vehicle_name=drone_name)
    time.sleep(0.01)
    client.armDisarm(True, vehicle_name=drone_name)
    time.sleep(0.01)
    client.setTrajectoryTrackerGains(airsim.TrajectoryTrackerGains().to_list(), vehicle_name=drone_name)
    time.sleep(0.01)

    # destroy all previous gates in map
    racing_utils.trajectory_utils.AllGatesDestroyer(client)

    # spawn red gates in appropriate locations
    # gate_poses = racing_utils.trajectory_utils.RedGateSpawner(client, num_gates=1, noise_amp=0)
    offset = [0, 0, -0]
    gate_poses = racing_utils.trajectory_utils.RedGateSpawnerCircle(client, num_gates=8, radius=8, radius_noise=3.0, height_range=[0, -3], track_offset=offset)

    # wait till takeoff complete
    vel_max = 5.0
    acc_max = 2.0

    time.sleep(1.0)

    # takeoff_position = airsim.Vector3r(25, 7, -1.5)
    # takeoff_orientation = airsim.Vector3r(.2, -0.9, 0)
    #
    # takeoff_position = airsim.Vector3r(25, -7, -1.5)
    # takeoff_orientation = airsim.Vector3r(-.2, 0.9, 0)

    takeoff_position = airsim.Vector3r(5.5, -4, -1.5+offset[2])
    takeoff_orientation = airsim.Vector3r(0.3, 0.9, 0)

    # takeoff_position = airsim.Vector3r(0, 0, -2)
    # takeoff_position = airsim.Vector3r(0, 0, 10)
    # takeoff_orientation = airsim.Vector3r(1, 0, 0)
    # client.plot_tf([takeoff_pose], duration=20.0, vehicle_name=drone_name)
    # client.moveOnSplineAsync([airsim.Vector3r(0, 0, -3)], vel_max=15.0, acc_max=5.0, vehicle_name=drone_name, viz_traj=True).join()
    client.moveOnSplineVelConstraintsAsync([takeoff_position], [takeoff_orientation], vel_max=vel_max, acc_max=acc_max, vehicle_name=drone_name, viz_traj=False).join()
    # client.moveOnSplineVelConstraintsAsync([airsim.Vector3r(1, 0, 8)], [airsim.Vector3r(1, 0, 0)], vel_max=vel_max, acc_max=acc_max, vehicle_name=drone_name, viz_traj=True)

    time.sleep(1.0)
    img_res = 64

    training_mode = 'latent'  # 'full' or 'latent'
    # training_mode = 'full'  # 'full' or 'latent' or 'reg'
    # training_mode = 'reg'  # 'full' or 'latent' or 'reg'

    # bc_weights_path = '/home/rb/data/model_outputs/bc_full/bc_model_100.ckpt'
    # feature_weights_path = ''

    # bc_weights_path = '/home/rb/data/model_outputs/bc_reg/bc_model_80.ckpt'
    # feature_weights_path = '/home/rb/data/model_outputs/reg/reg_model_25.ckpt'

    # bc_weights_path = '/home/rb/data/model_outputs/bc_unc/bc_model_100.ckpt'
    # feature_weights_path = '/home/rb/data/model_outputs/cmvae_unc/cmvae_model_45.ckpt'

    # bc_weights_path = '/home/rb/data/model_outputs/bc_con/bc_model_150.ckpt'
    # feature_weights_path = '/home/rb/data/model_outputs/cmvae_con/cmvae_model_40.ckpt'

    bc_weights_path = '/home/rb/data/model_outputs/bc_img/bc_model_100.ckpt'
    feature_weights_path = '/home/rb/data/model_outputs/cmvae_img/cmvae_model_45.ckpt'

    # bc_weights_path = '/home/rb/data/model_outputs/bc_real/bc_model_100.ckpt'
    # feature_weights_path = '/home/rb/data/model_outputs/cmvae_real/cmvae_model_40.ckpt'

    vel_regressor = vel_regressor.VelRegressor(regressor_type=training_mode, bc_weights_path=bc_weights_path, feature_weights_path=feature_weights_path)

    count = 0
    max_count = 50
    times_net = np.zeros((max_count,))
    times_loop = np.zeros((max_count,))
    while True:
        start_time = time.time()
        img_batch_1, cam_pos, cam_orientation = process_image(client, img_res)
        elapsed_time_net = time.time() - start_time
        times_net[count] = elapsed_time_net
        p_o_b = airsim.types.Pose(cam_pos, cam_orientation)
        vel_cmd = vel_regressor.predict_velocities(img_batch_1, p_o_b)
        # print(vel_cmd)
        # print('Before sending vel cmd')
        move_drone(client, vel_cmd)
        # print('After sending vel cmd')
        # time.sleep(0.05)
        elapsed_time_loop = time.time() - start_time
        times_loop[count] = elapsed_time_loop
        count = count + 1
        if count == max_count:
            count = 0
            avg_time = np.mean(times_net)
            avg_freq = 1.0/avg_time
            print('Avg network time over {} iterations: {} ms | {} Hz'.format(max_count, avg_time*1000, avg_freq))
            avg_time = np.mean(times_loop)
            avg_freq = 1.0 / avg_time
            print('Avg loop time over {} iterations: {} ms | {} Hz'.format(max_count, avg_time * 1000, avg_freq))
