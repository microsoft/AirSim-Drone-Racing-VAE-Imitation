from __future__ import division
import time
import numpy as np
import vel_regressor
import cv2
import math
import tensorflow as tf

import os, sys
import airsimdroneracingvae
import airsimdroneracingvae.types
import airsimdroneracingvae.utils

# import utils
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..')
sys.path.insert(0, import_path)
import racing_utils


###########################################

# DEFINE DEPLOYMENT META PARAMETERS

# policy options: bc_con, bc_unc, bc_img, bc_reg, bc_full
policy_type = 'bc_con'
gate_noise = 1.0

###########################################

def process_image(client, img_res):
    image_response = client.simGetImages([airsimdroneracingvae.ImageRequest('0', airsimdroneracingvae.ImageType.Scene, False, False)])[0]
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
    yaw_mode = airsimdroneracingvae.YawMode(is_rate=True, yaw_or_rate=vel_cmd[3]*180.0/np.pi)
    client.moveByVelocityAsync(vel_cmd[0], vel_cmd[1], vel_cmd[2], duration=0.1, yaw_mode=yaw_mode)


print(os.path.abspath(airsimdroneracingvae.__file__))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
# tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')
# tf.debugging.set_log_device_placement(True)


if __name__ == "__main__":
    # set airsim client
    client = airsimdroneracingvae.MultirotorClient()
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
    client.setTrajectoryTrackerGains(airsimdroneracingvae.TrajectoryTrackerGains().to_list(), vehicle_name=drone_name)
    time.sleep(0.01)

    # destroy all previous gates in map
    racing_utils.trajectory_utils.AllGatesDestroyer(client)

    # spawn red gates in appropriate locations
    # gate_poses = racing_utils.trajectory_utils.RedGateSpawner(client, num_gates=1, noise_amp=0)
    offset = [0, 0, -0]
    gate_poses = racing_utils.trajectory_utils.RedGateSpawnerCircle(client, num_gates=8, radius=8, radius_noise=gate_noise, height_range=[0, -gate_noise], track_offset=offset)

    # wait till takeoff complete
    vel_max = 5.0
    acc_max = 2.0

    time.sleep(1.0)

    # takeoff_position = airsimdroneracingvae.Vector3r(25, 7, -1.5)
    # takeoff_orientation = airsimdroneracingvae.Vector3r(.2, -0.9, 0)
    #
    # takeoff_position = airsimdroneracingvae.Vector3r(25, -7, -1.5)
    # takeoff_orientation = airsimdroneracingvae.Vector3r(-.2, 0.9, 0)

    takeoff_position = airsimdroneracingvae.Vector3r(5.5, -4, -1.5+offset[2])
    takeoff_orientation = airsimdroneracingvae.Vector3r(0.4, 0.9, 0)

    # takeoff_position = airsimdroneracingvae.Vector3r(0, 0, -2)
    # takeoff_position = airsimdroneracingvae.Vector3r(0, 0, 10)
    # takeoff_orientation = airsimdroneracingvae.Vector3r(1, 0, 0)
    # client.plot_tf([takeoff_pose], duration=20.0, vehicle_name=drone_name)
    # client.moveOnSplineAsync([airsimdroneracingvae.Vector3r(0, 0, -3)], vel_max=15.0, acc_max=5.0, vehicle_name=drone_name, viz_traj=True).join()
    client.moveOnSplineVelConstraintsAsync([takeoff_position], [takeoff_orientation], vel_max=vel_max, acc_max=acc_max, vehicle_name=drone_name, viz_traj=False).join()
    # client.moveOnSplineVelConstraintsAsync([airsimdroneracingvae.Vector3r(1, 0, 8)], [airsimdroneracingvae.Vector3r(1, 0, 0)], vel_max=vel_max, acc_max=acc_max, vehicle_name=drone_name, viz_traj=True)

    time.sleep(1.0)
    img_res = 64

    if policy_type == 'bc_con':
        training_mode = 'latent'
        latent_space_constraints = True
        bc_weights_path = '/home/rb/all_files/model_outputs/bc_con/bc_model_150.ckpt'
        feature_weights_path = '/home/rb/all_files/model_outputs/cmvae_con/cmvae_model_40.ckpt'
    elif policy_type == 'bc_unc':
        training_mode = 'latent'
        latent_space_constraints = False
        bc_weights_path = '/home/rb/all_files/model_outputs/bc_unc/bc_model_150.ckpt'
        feature_weights_path = '/home/rb/all_files/model_outputs/cmvae_unc/cmvae_model_45.ckpt'
    elif policy_type == 'bc_img':
        training_mode = 'latent'
        latent_space_constraints = True
        bc_weights_path = '/home/rb/all_files/model_outputs/bc_img/bc_model_100.ckpt'
        feature_weights_path = '/home/rb/all_files/model_outputs/cmvae_img/cmvae_model_45.ckpt'
    elif policy_type == 'bc_reg':
        training_mode = 'reg'
        latent_space_constraints = True
        bc_weights_path = '/home/rb/all_files/model_outputs/bc_reg/bc_model_80.ckpt'
        feature_weights_path = '/home/rb/all_files/model_outputs/reg/reg_model_25.ckpt'
    elif policy_type == 'bc_full':
        training_mode = 'full'
        latent_space_constraints = True
        bc_weights_path = '/home/rb/all_files/model_outputs/bc_full/bc_model_120.ckpt'
        feature_weights_path = None

    vel_regressor = vel_regressor.VelRegressor(regressor_type=training_mode, bc_weights_path=bc_weights_path,
                                               feature_weights_path=feature_weights_path,
                                               latent_space_constraints=latent_space_constraints)

    count = 0
    max_count = 50
    times_net = np.zeros((max_count,))
    times_loop = np.zeros((max_count,))
    while True:
        start_time = time.time()
        img_batch_1, cam_pos, cam_orientation = process_image(client, img_res)
        elapsed_time_net = time.time() - start_time
        times_net[count] = elapsed_time_net
        p_o_b = airsimdroneracingvae.types.Pose(cam_pos, cam_orientation)
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
