import os
import sys
from scipy.spatial.transform import Rotation
import math
from airsimdroneracingvae.utils import to_eularian_angles, to_quaternion
import numpy as np
from airsimdroneracingvae.types import Pose, Vector3r, Quaternionr

def interp_vector(a, b, n):
    delta = (b-a)/(n-1)
    list_vecs = []
    for i in range(n):
        new_vec = a+delta*i
        list_vecs.append(new_vec)
    return np.asarray(list_vecs)

def randomQuadPose(x_range, y_range, z_range, yaw_range, pitch_range, roll_range):
    x = randomSample(x_range)
    y = randomSample(y_range)
    z = randomSample(z_range)
    yaw = randomSample(yaw_range)
    pitch = randomSample(pitch_range)
    roll = randomSample(roll_range)
    q = Rotation.from_euler('ZYX', [yaw, pitch, roll])  # capital letters denote intrinsic rotation (lower case would be extrinsic)
    q = q.as_quat()
    t_o_b = Vector3r(x,y,z)
    q_o_b = Quaternionr(q[0], q[1], q[2], q[3])
    return Pose(t_o_b, q_o_b), yaw

def randomSample(value_range):
    return (value_range[1] - value_range[0])*np.random.random() + value_range[0]

def randomGatePose(p_o_b, phi_base, r_range, cam_fov, correction):
    gate_ok = False
    while not gate_ok:
        # create translation of gate
        r = randomSample(r_range)
        alpha = cam_fov/180.0*np.pi/2.0  # alpha is half of fov angle
        theta_range = [-alpha, alpha]
        theta = randomSample(theta_range)
        # need to make projection on geodesic curve! not equal FOV in theta and psi
        alpha_prime = np.arctan(np.cos(np.abs(theta)))
        psi_range = [-alpha_prime, alpha_prime]
        psi_range = [x * correction for x in psi_range]
        psi = randomSample(psi_range) + np.pi/2.0
        # get relative vector in the base frame coordinates
        t_b_g_body = polarTranslation(r, theta, psi)

        # transform relative vector from base frame to the world frame
        t_b_g = convert_t_body_2_world(t_b_g_body, p_o_b.orientation)

        # get the gate coord in world coordinates from origin
        t_o_g = p_o_b.position + t_b_g

        # check if gate is at least half outside the ground
        if t_o_g.z_val >= 0.0:
            continue

        # create rotation of gate
        eps = 0  # np.pi/15.0
        phi_rel_range = [-np.pi + eps, 0 - eps]
        phi_rel = randomSample(phi_rel_range)
        phi_quad_ref = get_yaw_base(p_o_b)
        phi_gate = phi_quad_ref + phi_rel
        rot_gate = Rotation.from_euler('ZYX', [phi_gate, 0, 0])
        q = rot_gate.as_quat()
        p_o_g = Pose(t_o_g, Quaternionr(q[0], q[1], q[2], q[3]))

        return p_o_g, r, theta, psi, phi_rel

def debugRelativeOrientation(p_o_b, p_o_g, phi_rel):
    phi_quad_ref = get_yaw_base(p_o_b)
    phi_gate = phi_quad_ref + phi_rel
    rot_gate = Rotation.from_euler('ZYX', [phi_gate, 0, 0])
    q = rot_gate.as_quat()
    p_o_g = Pose(p_o_g.position, Quaternionr(q[0], q[1], q[2], q[3]))
    return p_o_g

def debugGatePoses(p_o_b, r, theta, psi):
    # get relative vector in the base frame coordinates
    t_b_g_body = polarTranslation(r, theta, psi)
    # transform relative vector from base frame to the world frame
    t_b_g = convert_t_body_2_world(t_b_g_body, p_o_b.orientation)
    # get the gate coord in world coordinates from origin
    t_o_g = p_o_b.position + t_b_g
    # check if gate is at least half outside the ground
    # create rotation of gate
    phi_quad_ref = np.arctan2(p_o_b.position.y_val, p_o_b.position.x_val)
    phi_rel = np.pi/2
    phi_gate = phi_quad_ref + phi_rel
    rot_gate = Rotation.from_euler('ZYX', [phi_gate, 0, 0])
    q = rot_gate.as_quat()
    p_o_g = Pose(t_o_g, Quaternionr(q[0], q[1], q[2], q[3]))
    return p_o_g, r, theta, psi, phi_rel

def polarTranslation(r, theta, psi):
    # follow math convention for polar coordinates
    # r: radius
    # theta: azimuth (horizontal)
    # psi: vertical
    x = r * np.cos(theta) * np.sin(psi)
    y = r * np.sin(theta) * np.sin(psi)
    z = r * np.cos(psi)
    return Vector3r(x, y, z)

def convert_t_body_2_world(t_body, q_o_b):
    rotation = Rotation.from_quat([q_o_b.x_val, q_o_b.y_val, q_o_b.z_val, q_o_b.w_val])
    t_body_np = [t_body.x_val, t_body.y_val, t_body.z_val]
    t_world_np = rotation.apply(t_body_np)
    t_world = Vector3r(t_world_np[0], t_world_np[1], t_world_np[2])
    return t_world

def get_yaw_base(p_o_b):
    q_o_b = p_o_b.orientation
    rotation = Rotation.from_quat([q_o_b.x_val, q_o_b.y_val, q_o_b.z_val, q_o_b.w_val])
    euler_angles = rotation.as_euler('ZYX')
    return euler_angles[0]

# this is utility function to get a velocity constraint which can be passed to moveOnSplineVelConstraints() 
# the "scale" parameter scales the gate facing vector accordingly, thereby dictating the speed of the velocity constraint
def get_gate_facing_vector_from_quaternion(airsim_quat, direction, scale=1.0,):
    # convert gate quaternion to rotation matrix. 
    # ref: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion; https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
    q = np.array([airsim_quat.w_val, airsim_quat.x_val, airsim_quat.y_val, airsim_quat.z_val], dtype=np.float64)
    n = np.dot(q, q)
    if n < np.finfo(float).eps:
        if direction == 0:
            return airsimdroneracingvae.Vector3r(0.0, 1.0, 0.0)
        else:
            return airsimdroneracingvae.Vector3r(0.0, -1.0, 0.0)

    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    rotation_matrix = np.array([[1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
                                [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
                                [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])
    gate_facing_vector = rotation_matrix[:,1]
    if direction == 0:
        return airsimdroneracingvae.Vector3r(scale * gate_facing_vector[0], scale * gate_facing_vector[1], scale * gate_facing_vector[2])
    else:
        return airsimdroneracingvae.Vector3r(-scale * gate_facing_vector[0], -scale * gate_facing_vector[1], scale * gate_facing_vector[2])

def getGatePoseWorld(p_o_b, r, theta, psi, phi_rel):
    # get relative vector in the base frame coordinates
    t_b_g_body = polarTranslation(r, theta, psi)
    # transform relative vector from base frame to the world frame
    t_b_g = convert_t_body_2_world(t_b_g_body, p_o_b.orientation)
    # get the gate coord in world coordinates from origin
    t_o_g = p_o_b.position + t_b_g
    # create rotation of gate
    phi_quad_ref = get_yaw_base(p_o_b)
    phi_gate = phi_quad_ref + phi_rel
    rot_gate = Rotation.from_euler('ZYX', [phi_gate, 0, 0])
    q = rot_gate.as_quat()
    p_o_g = Pose(t_o_g, Quaternionr(q[0], q[1], q[2], q[3]))
    return p_o_g