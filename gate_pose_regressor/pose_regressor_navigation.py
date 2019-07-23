from __future__ import division
import random
import airsim
import airsim.types
import airsim.utils
import math
import copy
import time
import numpy as np
import threading
import pdb
import gate_regressor

import os,sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
airsim_path = os.path.join(curr_dir, '..', 'airsim')
sys.path.insert(0, airsim_path)
import setup_path


def takeoff_with_moveOnSpline(client, takeoff_height=1.2, vel_max=15.0, acc_max=5.0):
    client.moveOnSplineAsync([airsim.Vector3r(0, 0, -takeoff_height)], vel_max=vel_max, acc_max=acc_max, vehicle_name='drone_0').join()


if __name__ == "__main__":
    # set airsim client
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, vehicle_name='drone_0')
    time.sleep(0.01)

    img_resolution = 96
    path_weights = '/home/rb/catkin_ws/src/AutonomousDrivingCookbook/AirSimE2EDeepLearning/output_reg2/regressor_model_100.ckpt'
    gate_regressor = gate_regressor.GateRegressor()

    #todo: setup client and quadrotor properly for takeoff

    while True:
        image_response = client.simGetImages([airsim.ImageRequest('front_center_custom', airsim.ImageType.Scene, False, False)])[0]
        cam_pos = image_response.camera_position
        cam_orientation = image_response.camera_orientation
        gate_pose = model_predict_gate_pose(image_response.image_data_uint8, cam_pos, cam_orientation)
        #todo: adjust spline points to .5m in front and behind gates after this works
        client.moveOnSplineAsync([gate_pose.position], vel_max=vel_max, acc_max=acc_max, vehicle_name=self.drone_name).join()
