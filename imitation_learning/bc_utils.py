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
import glob
import os
from PIL import Image
import tensorflow
from sklearn.model_selection import train_test_split
from scipy.spatial.transform import Rotation

def convert_vel_base2world(p_o_b, vx, vy, vz, v_yaw):
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


def RedGateSpawner(client):
    for idx in range(10):
        pose = Pose(Vector3r(10+idx*6, (np.random.random()-0.5)*5.0, 10.0), Quaternionr(0.0, 0.0, 0.707, 0.707))
        client.simSpawnObject("gate_"+str(idx), "RedGate16x16", pose, 1.5)


def v_body_to_world(v_body, p_o_b):
    q = np.zeros((4,))
    q[0] = p_o_b.orientation.x_val
    q[1] = p_o_b.orientation.y_val
    q[2] = p_o_b.orientation.z_val
    q[3] = p_o_b.orientation.w_val
    rotation = Rotation.from_quat(q)
    v_world = rotation.apply(v_body[:3])
    # copy the yaw velocity over
    v_world = np.append(v_world, v_body[3])
    return v_world


def normalize_v(v):
    # normalization of velocities from whatever to [-1, 1] range
    v_x_range = [-1, 7]
    v_y_range = [-3, 3]
    v_z_range = [-3, 3]
    v_yaw_range = [-1, 1]
    if len(v.shape) == 1:
        # means that it's a 1D vector of velocities
        v[0] = 2.0 * (v[0] - v_x_range[0]) / (v_x_range[1] - v_x_range[0]) - 1.0
        v[1] = 2.0 * (v[1] - v_y_range[0]) / (v_y_range[1] - v_y_range[0]) - 1.0
        v[2] = 2.0 * (v[2] - v_z_range[0]) / (v_z_range[1] - v_z_range[0]) - 1.0
        v[3] = 2.0 * (v[3] - v_yaw_range[0]) / (v_yaw_range[1] - v_yaw_range[0]) - 1.0
    elif len(v.shape) == 2:
        # means that it's a 2D vector of velocities
        v[:, 0] = 2.0 * (v[:, 0] - v_x_range[0]) / (v_x_range[1] - v_x_range[0]) - 1.0
        v[:, 1] = 2.0 * (v[:, 1] - v_y_range[0]) / (v_y_range[1] - v_y_range[0]) - 1.0
        v[:, 2] = 2.0 * (v[:, 2] - v_z_range[0]) / (v_z_range[1] - v_z_range[0]) - 1.0
        v[:, 3] = 2.0 * (v[:, 3] - v_yaw_range[0]) / (v_yaw_range[1] - v_yaw_range[0]) - 1.0
    else:
        raise Exception('Error in data format of V shape: {}'.format(v.shape))
    return v


def de_normalize_v(v):
    # normalization of velocities from [-1, 1] range to whatever
    v_x_range = [-1, 7]
    v_y_range = [-3, 3]
    v_z_range = [-3, 3]
    v_yaw_range = [-1, 1]
    if len(v.shape) == 1:
        # means that it's a 1D vector of velocities
        v[0] = (v[0] + 1.0) / 2.0 * (v_x_range[1] - v_x_range[0]) + v_x_range[0]
        v[1] = (v[1] + 1.0) / 2.0 * (v_y_range[1] - v_y_range[0]) + v_y_range[0]
        v[2] = (v[2] + 1.0) / 2.0 * (v_z_range[1] - v_z_range[0]) + v_z_range[0]
        v[3] = (v[3] + 1.0) / 2.0 * (v_yaw_range[1] - v_yaw_range[0]) + v_yaw_range[0]
    elif len(v.shape) == 2:
        # means that it's a 2D vector of velocities
        v[:, 0] = (v[:, 0] + 1.0) / 2.0 * (v_x_range[1] - v_x_range[0]) + v_x_range[0]
        v[:, 1] = (v[:, 1] + 1.0) / 2.0 * (v_y_range[1] - v_y_range[0]) + v_y_range[0]
        v[:, 2] = (v[:, 2] + 1.0) / 2.0 * (v_z_range[1] - v_z_range[0]) + v_z_range[0]
        v[:, 3] = (v[:, 3] + 1.0) / 2.0 * (v_yaw_range[1] - v_yaw_range[0]) + v_yaw_range[0]
    else:
        raise Exception('Error in data format of V shape: {}'.format(v.shape))
    return v


def create_dataset_txt(data_dir, batch_size, res, num_channels):
    vel_table = np.loadtxt(data_dir + '/proc_vel.txt', delimiter=',').astype(np.float32)
    with open(data_dir + '/proc_images.txt') as f:
        img_table = f.read().splitlines()

    # sanity check
    if vel_table.shape[0] != len(img_table):
        raise Exception('Number of images ({}) different than number of entries in table ({}): '.format(len(img_table), vel_table.shape[0]))

    images_list = []
    for img_name in img_table:
        im = Image.open(img_name).resize((res, res), Image.BILINEAR)
        im = np.array(im)/255.0*2 - 1.0  # convert to the -1 -> 1 scale
        # TODO: figure out why there's a 4th channel in dataset
        im = im[:,:,:3]
        images_list.append(im)
    images_np = np.array(images_list).astype(np.float32)

    # print some useful statistics and normalize distances
    print("Num samples: {}".format(vel_table.shape[0]))
    print("Average vx: {}".format(np.mean(vel_table[:, 0])))
    print("Average vy: {}".format(np.mean(vel_table[:, 1])))
    print("Average vz: {}".format(np.mean(vel_table[:, 2])))
    print("Average vyaw: {}".format(np.mean(vel_table[:, 3])))

    # normalize the values of velocities to the [-1, 1] range
    vel_table = normalize_v(vel_table)

    img_train, img_test, dist_train, dist_test = train_test_split(images_np, vel_table, test_size=0.1, random_state=42)

    # convert to tf format dataset and prepare batches
    ds_train = tensorflow.data.Dataset.from_tensor_slices((img_train, dist_train)).batch(batch_size)
    ds_test = tensorflow.data.Dataset.from_tensor_slices((img_test, dist_test)).batch(batch_size)

    return ds_train, ds_test
