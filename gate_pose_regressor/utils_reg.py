from __future__ import division
import airsim
from airsim.utils import to_eularian_angles, to_quaternion
import numpy as np
from airsim.types import Pose, Vector3r, Quaternionr
import nav_msgs.msg
import geometry_msgs.msg
import tf2_ros
import tf_conversions
import tf2_msgs
import tf2_py
import tf
import tf.msg
from tf import transformations
import rospy
import time
import random
import math

def polarTranslation(r, theta, psi):
    # follow math convention for polar coordinates
    # r: radius
    # theta: azimuth (horizontal)
    # psi: vertical
    x = r * np.cos(theta) * np.sin(psi)
    y = r * np.sin(theta) * np.sin(psi)
    z = r * np.cos(psi)
    return Vector3r(x, y, z)


def convert_gate_base2world(p_o_b, t_b_g, phi_rel):
    # transform relative vector from base frame to the world frame
    m = geometry_msgs.msg.TransformStamped()
    m.header.frame_id = 'world_frame'
    m.header.stamp = rospy.Time(10)
    m.child_frame_id = 'base_frame'
    m.transform.translation.x = p_o_b.position
    m.transform.rotation.x = p_o_b.orientation.x_val
    m.transform.rotation.y = p_o_b.orientation.y_val
    m.transform.rotation.z = p_o_b.orientation.z_val
    m.transform.rotation.w = p_o_b.orientation.w_val
    m.transform.translation.x = p_o_b.position.x_val
    m.transform.translation.y = p_o_b.position.y_val
    m.transform.translation.z = p_o_b.position.z_val

    point = geometry_msgs.msg.PointStamped()
    point.point.x = t_b_g.x_val
    point.point.y = t_b_g.y_val
    point.point.z = t_b_g.z_val
    point.header.frame_id = 'base_frame'
    point.header.stamp = rospy.Time(10)  # set an arbitrary time instance

    t = tf.TransformerROS()
    t.setTransform(m)

    # finally we get the gate coord in world coordinates
    point_converted = t.transformPoint('world_frame', point)
    t_o_g = Vector3r(point_converted.point.x, point_converted.point.y, point_converted.point.z)  # now as a world coord vector

    # create rotation of gate
    # find vector t_b_g in world coordinates
    t_b_g = t_o_g - p_o_b.position
    phi_quad_ref = np.arctan2(t_b_g.y_val, t_b_g.x_val)
    phi_gate = phi_quad_ref + phi_rel
    q = transformations.quaternion_from_euler(phi_gate, 0, 0, axes='rzyx')
    p_o_g = Pose(t_o_g, Quaternionr(q[0], q[1], q[2], q[3]))
    return p_o_g


def MoveCheckeredGates(client):
    gate_names_sorted = sorted(client.simListSceneObjects("Gate.*"))
    # gate_names_sorted_bad is ['Gate0', 'Gate10_21', 'Gate11_23', 'Gate1_3', 'Gate2_5', 'Gate3_7', 'Gate4_9', 'Gate5_11', 'Gate6_13', 'Gate7_15', 'Gate8_17', 'Gate9_19']
    # number after underscore is unreal garbage
    pose_far = Pose(Vector3r(0,0,1), Quaternionr())
    for gate in gate_names_sorted:
        client.simSetObjectPose(gate, pose_far)
        # time.sleep(0.05)


def RedGateSpawner(client, noise_amp):
    for idx in range(10):
        noise = (np.random.random()-0.5)*noise_amp
        pose = Pose(Vector3r(10+idx*6, noise*5.0, 10.0), Quaternionr(0.0, 0.0, 0.707, 0.707))
        client.simSpawnObject("gate_"+str(idx), "RedGate16x16", pose, 1.5)


def RedGateSpawnerCircle(client):
    num_gates = 10
    track = generate_gate_poses(num_gates=num_gates, race_course_radius=30)
    for idx in range(num_gates):
        client.simSpawnObject("gate_" + str(idx), "RedGate16x16", track[idx], 1.5)


def generate_gate_poses(num_gates, race_course_radius, type_of_segment="circle"):
    if type_of_segment == "circle":
        (x_t, y_t, z_t) = tuple([generate_circle(i, num_gates, race_course_radius) for i in range(3)])
        # todo unreadable code
        # todo un-hardcode
        gate_poses = [\
                        airsim.Pose(\
                        airsim.Vector3r((x_t[t_i][0] - x_t[0][0] - 4.0), (y_t[t_i][0] - y_t[0][0] - 4.0), random.uniform(-5.0, -9.0)),\
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


def generate_circle(i, num_gates, race_course_radius):
    ts = [t / (num_gates) for t in range(0, num_gates)]
    samples = [0 for t in ts]
    derivatives = [0 for t in ts]
    min_radius = race_course_radius + 4.0
    max_radius = race_course_radius - 4.0
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