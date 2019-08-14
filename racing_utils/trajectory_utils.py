from __future__ import division
import numpy as np
import random
import math

import os, sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
airsim_path = os.path.join(curr_dir, '..', 'airsim')
sys.path.insert(0, airsim_path)
import setup_path
import airsim
import airsim.types
import airsim.utils
from airsim.types import Pose, Vector3r, Quaternionr
from airsim.utils import to_eularian_angles, to_quaternion


def MoveCheckeredGates(client):
    gate_names_sorted = sorted(client.simListSceneObjects("Gate.*"))
    # gate_names_sorted_bad is ['Gate0', 'Gate10_21', 'Gate11_23', 'Gate1_3', 'Gate2_5', 'Gate3_7', 'Gate4_9', 'Gate5_11', 'Gate6_13', 'Gate7_15', 'Gate8_17', 'Gate9_19']
    # number after underscore is unreal garbage
    pose_far = Pose(Vector3r(0,0,1), Quaternionr())
    for gate in gate_names_sorted:
        client.simSetObjectPose(gate, pose_far)
        # time.sleep(0.05)


def AllGatesDestroyer(client):
    [client.simDestroyObject(gate_object) for gate_object in client.simListSceneObjects(".*[Gg]ate.*")]


def RedGateSpawner(client, num_gates, noise_amp):
    gate_poses=[]
    for idx in range(num_gates):
        noise = (np.random.random()-0.5)*noise_amp
        pose = Pose(Vector3r(10+idx*9, noise*5.0, 10.0), Quaternionr(0.0, 0.0, 0.707, 0.707))
        client.simSpawnObject("gate_"+str(idx), "RedGate16x16", pose, 1.5)
        gate_poses.append(pose)
    return gate_poses


def RedGateSpawnerCircle(client, num_gates, radius, radius_noise, height_range):
    track = generate_gate_poses(num_gates=num_gates, race_course_radius=radius, radius_noise=radius_noise, height_range=height_range)
    for idx in range(num_gates):
        client.simSpawnObject("gate_" + str(idx), "RedGate16x16", track[idx], 1.5)


def generate_gate_poses(num_gates, race_course_radius, radius_noise, height_range, type_of_segment="circle"):
    if type_of_segment == "circle":
        (x_t, y_t, z_t) = tuple([generate_circle(i, num_gates, race_course_radius, radius_noise) for i in range(3)])
        # todo unreadable code
        # todo un-hardcode
        # airsim.Vector3r((x_t[t_i][0] - x_t[0][0]), (y_t[t_i][0] - y_t[0][0]), random.uniform(height_range[0], height_range[1])), \
        gate_poses = [\
                        airsim.Pose(\
                        airsim.Vector3r((x_t[t_i][0]), (y_t[t_i][0]), random.uniform(height_range[0], height_range[1])), \
                        quaternionFromUnitGradient(x_t[t_i][1], y_t[t_i][1], z_t[t_i][1])\
                      )\
                    for t_i in range(num_gates)]
    # elif type_of_segment == "cubic":
    return gate_poses


def quaternionFromUnitGradient(dx_dt, dy_dt, dz_dt):
    default_gate_facing_vector = type("", (), dict(x=0, y=1, z=0))()
    r0 = default_gate_facing_vector
    q = airsim.Quaternionr(
            r0.y * dz_dt - r0.z * dy_dt,
            r0.z * dx_dt - r0.x * dz_dt,
            r0.x * dy_dt - r0.y * dx_dt,
            math.sqrt((r0.x**2 + r0.y**2 + r0.z**2) * (dx_dt**2 + dy_dt**2 + dz_dt**2)) + (r0.x * dx_dt + r0.y * dy_dt + r0.z * dz_dt)
        )
    #Normalize
    length = q.get_length()
    if (length == 0.0):
        q.w_val = 1.0
    else:
        q.w_val /= length
        q.x_val /= length
        q.y_val /= length
        q.z_val /= length
    return q


def generate_circle(i, num_gates, race_course_radius, radius_amp=4.0):
    ts = [t / (num_gates) for t in range(0, num_gates)]
    samples = [0 for t in ts]
    derivatives = [0 for t in ts]
    min_radius = race_course_radius + radius_amp
    max_radius = race_course_radius - radius_amp
    max_radius_delta = 5.0
    radius_list = [random.uniform(min_radius, max_radius) for t in ts]
    # not a circle, but hey it's random-ish. and the wrong derivative actually make the track challenging
    # come back again later.
    if i == 0:
        for (idx, t) in enumerate(ts):
            radius = radius_list[idx]
            if idx > 0:
                radius = np.clip(radius, radius_list[idx-1] - max_radius_delta, radius_list[idx-1] + max_radius_delta)
                radius = np.clip(radius, 0.0, radius)
            samples[idx] = radius * math.cos(2.*math.pi * t)
            derivatives[idx] = radius * -math.sin(2.*math.pi * t)
    elif i == 1:
        for (idx, t) in enumerate(ts):
            radius = radius_list[idx]
            if idx > 0:
                radius = np.clip(radius, radius_list[idx-1] - max_radius_delta, radius_list[idx-1] + max_radius_delta)
                radius = np.clip(radius, 0.0, radius)
            samples[idx] = radius * math.sin(2.*math.pi * t)
            derivatives[idx] = radius * math.cos(2.*math.pi * t)
    else:
        for (idx, t) in enumerate(ts):
            samples[idx] = 0.
            derivatives[idx] = 0.
    return list(zip(samples, derivatives))