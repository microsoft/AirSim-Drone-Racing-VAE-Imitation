from __future__ import division
import time
import numpy as np
import gate_regressor
import cv2
import math

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


def move_drone(client, gate_pose):
    # client.enableApiControl(True, vehicle_name=drone_name)
    # client.moveOnSplineAsync([gate_pose.position], vel_max=vel_max, acc_max=acc_max, vehicle_name=drone_name)
    gate_vector = racing_utils.geom_utils.get_gate_facing_vector_from_quaternion(gate_pose.orientation, scale=1.0)
    # client.plot_tf([airsim.Pose(gate_pose.position + gate_vector, gate_pose.orientation)], duration=20.0, vehicle_name=drone_name)
    # client.plot_tf([gate_pose], duration=20.0, vehicle_name=drone_name)
    # client.moveOnSplineAsync([gate_pose.position], [gate_vector], vel_max=vel_max, acc_max=acc_max, vehicle_name=drone_name, viz_traj=True)
    # client.moveOnSplineVelConstraintsAsync([gate_pose.position], [gate_vector], add_curr_odom_position_constraint=True, add_curr_odom_velocity_constraint=True, vel_max=vel_max, acc_max=acc_max, vehicle_name=drone_name, viz_traj=True)
    client.moveOnSplineAsync([gate_pose.position],
                             add_curr_odom_position_constraint=True,
                             add_curr_odom_velocity_constraint=True,
                             vel_max=vel_max, acc_max=acc_max,
                             vehicle_name=drone_name, viz_traj=False)
    # client.moveOnSplineVelConstraintsAsync([gate_pose.position-gate_vector],
    #                                        [gate_vector],
    #                                        add_curr_odom_position_constraint=True,
    #                                        add_curr_odom_velocity_constraint=True, vel_max=vel_max, acc_max=acc_max,
    #                                        vehicle_name=drone_name, viz_traj=True)


print(os.path.abspath(airsim.__file__))

if __name__ == "__main__":
    # set airsim client
    client = airsim.MultirotorClient()
    client.confirmConnection()
    # client.simLoadLevel('Soccer_Field_Easy')
    client.simLoadLevel('Soccer_Field_Medium')
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
    gate_poses = racing_utils.trajectory_utils.RedGateSpawnerCircle(client, num_gates=13, radius=20, radius_noise=0.0, height_range=[10, 11])

    # wait till takeoff complete
    vel_max = 3.0
    acc_max = 3.0

    takeoff_position = airsim.Vector3r(20, -2, 10)
    takeoff_orientation = airsim.Vector3r(0, -1, 0)
    # takeoff_position = airsim.Vector3r(0, 0, 10)
    # takeoff_orientation = airsim.Vector3r(1, 0, 0)
    # client.plot_tf([takeoff_pose], duration=20.0, vehicle_name=drone_name)
    # client.moveOnSplineAsync([airsim.Vector3r(0, 0, -3)], vel_max=15.0, acc_max=5.0, vehicle_name=drone_name, viz_traj=True).join()
    client.moveOnSplineVelConstraintsAsync([takeoff_position], [takeoff_orientation], vel_max=vel_max, acc_max=acc_max, vehicle_name=drone_name, viz_traj=True).join()
    # client.moveOnSplineVelConstraintsAsync([airsim.Vector3r(1, 0, 8)], [airsim.Vector3r(1, 0, 0)], vel_max=vel_max, acc_max=acc_max, vehicle_name=drone_name, viz_traj=True)

    time.sleep(1.0)
    img_res = 64

    # path_weights = '/home/rb/data/model_outputs/reg_6/reg_model_10.ckpt'
    path_weights = '/home/rb/data/model_outputs/cmvae_9/cmvae_model_20.ckpt'
    gate_regressor = gate_regressor.GateRegressor(regressor_type='cmvae', path_weights=path_weights)

    while True:
        img_batch_1, cam_pos, cam_orientation = process_image(client, img_res)
        p_o_b = airsim.types.Pose(cam_pos, cam_orientation)
        gate_pose = gate_regressor.predict_gate_pose(img_batch_1, p_o_b)
        print(gate_pose.position)
        print('Before move spline')
        move_drone(client, gate_pose)
        print('Done move spline')
        time.sleep(1.0)
