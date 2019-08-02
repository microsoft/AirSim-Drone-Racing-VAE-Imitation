from __future__ import division

import os, sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
airsim_path = os.path.join(curr_dir, '..', 'airsim')
sys.path.insert(0, airsim_path)
import setup_path
import airsim
import airsim.types
import airsim.utils
import time
import numpy as np
import gate_regressor
import utils_reg

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
    client.setTrajectoryTrackerGains(airsim.TrajectoryTrackerGains(), vehicle_name=drone_name)
    time.sleep(0.01)

    # utils_reg.MoveCheckeredGates(client)
    utils_reg.RedGateSpawner(client)

    # wait till takeoff complete
    time.sleep(0.2)
    time.sleep(0.05)
    client.moveOnSplineAsync([airsim.Vector3r(0, 0, 10)], vel_max=15.0, acc_max=5.0, vehicle_name=drone_name).join()

    time.sleep(1.0)

    img_resolution = 96
    vel_max = 15.0
    acc_max = 5.00
    path_weights = '/home/rb/data/model_outputs/reg_1/reg_model_185.ckpt'
    gate_regressor = gate_regressor.GateRegressor(img_resolution, path_weights)

    while True:
        image_response = client.simGetImages([airsim.ImageRequest('front_center_custom', airsim.ImageType.Scene, False, False)])[0]
        img_1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)  # get numpy array
        img_rgb = img_1d.reshape(image_response.height, image_response.width, 3).astype(np.float32)  # reshape array to 4 channel image array H X W X 3
        img_batch_1 = np.reshape(img_rgb, (-1,)+img_rgb.shape)
        # print('size img = {}'.format(img_batch_1.shape))
        cam_pos = image_response.camera_position
        cam_orientation = image_response.camera_orientation
        p_o_b = airsim.types.Pose(cam_pos, cam_orientation)
        gate_pose = gate_regressor.predict_gate_pose(img_batch_1, p_o_b)
        #todo: adjust spline points to .5m in front and behind gates after this works
        print(gate_pose.position)
        print('Before move spline')
        client.moveByVelocityAsync()
        client.moveOnSplineAsync([gate_pose.position], vel_max=vel_max, acc_max=acc_max, vehicle_name=drone_name)
        print('Done move spline')
        time.sleep(1.0)
