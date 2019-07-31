
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

print(os.path.abspath(airsim.__file__))

if True:
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.simLoadLevel('Soccer_Field_Easy')
    time.sleep(2)
    # should match the names in settings.json
    drone_name = "drone_0"

    client.enableApiControl(True, vehicle_name=drone_name)
    time.sleep(0.01)
    client.armDisarm(True, vehicle_name=drone_name)
    time.sleep(0.01)
    client.setTrajectoryTrackerGains(airsim.TrajectoryTrackerGains(), vehicle_name=drone_name)
    time.sleep(0.01)

    # wait till takeoff complete
    time.sleep(0.2)
    time.sleep(0.05)
    client.moveOnSplineAsync([airsim.Vector3r(0, 0, -0.3)], vel_max=15.0, acc_max=5.0, vehicle_name=drone_name).join()

