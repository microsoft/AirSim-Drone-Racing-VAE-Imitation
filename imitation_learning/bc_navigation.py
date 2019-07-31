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
import bc_regressor
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
    noise_amp = 0.0
    # utils_reg.RedGateSpawner(client, noise_amp)
    utils_reg.RedGateSpawnerCircle(client)
    print('Done with red gates')

    # wait till takeoff complete
    time.sleep(0.2)
    time.sleep(0.05)
    print('Moving to position ... ')
    # client.moveOnSplineAsync([airsim.Vector3r(0, 0, 10)], vel_max=15.0, acc_max=5.0, vehicle_name=drone_name).join()
    client.moveOnSplineAsync([airsim.Vector3r(0, 0, -15)], vel_max=50.0, acc_max=50.0, vehicle_name=drone_name).join()

    time.sleep(1.0)
    print('... Position achieved')

    img_resolution = 120
    path_weights = '/home/rb/data/il_datasets/il_2/output_bc_4/bc_model_300.ckpt'
    vel_regressor = bc_regressor.VelRegressor(img_resolution, path_weights)

    while True:
        start = time.time()
        # image_response = client.simGetImages([airsim.ImageRequest('front_center_custom', airsim.ImageType.Scene, False, False)])[0]
        image_response = client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
        img_1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)  # get numpy array
        img_rgb = img_1d.reshape(image_response.height, image_response.width, 3).astype(np.float32)  # reshape array to 4 channel image array H X W X 3
        img_batch_1 = np.reshape(img_rgb, (-1,)+img_rgb.shape)
        # print('size img = {}'.format(img_batch_1.shape))
        cam_pos = image_response.camera_position
        cam_orientation = image_response.camera_orientation
        p_o_b = airsim.types.Pose(cam_pos, cam_orientation)
        vel_cmd = vel_regressor.predict_velocities(img_batch_1, p_o_b)
        print(vel_cmd)
        vel_cmd = vel_cmd * 1
        yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=vel_cmd[3])
        client.moveByVelocityAsync(vel_cmd[0], vel_cmd[1], vel_cmd[2], duration=0.1, yaw_mode=yaw_mode)
        end = time.time()
        print(end - start)
        # time.sleep(1.0)
