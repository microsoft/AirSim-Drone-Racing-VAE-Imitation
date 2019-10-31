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


def normalizeQuat(quat):
    length = quat.get_length()
    if length < 1:
        print('stop')
    quat.w_val /= length
    quat.x_val /= length
    quat.y_val /= length
    quat.z_val /= length
    return quat


def randomSample(value_range):
    return (value_range[1] - value_range[0])*np.random.random() + value_range[0]


def tfQuat2msgQuat(tf_quat):
    msg_quat = geometry_msgs.msg.Quaternion()
    msg_quat.x = tf_quat[0]
    msg_quat.y = tf_quat[1]
    msg_quat.z = tf_quat[2]
    msg_quat.w = tf_quat[3]


def rosPose2airsimPose(ros_pose):
    t = Vector3r(ros_pose.position.x, ros_pose.position.y, ros_pose.position.z)
    q = Quaternionr(ros_pose.orientation.x, ros_pose.orientation.y, ros_pose.orientation.z, ros_pose.orientation.w)
    return Pose(t, q)


def airsimPose2rosPose(airsim_pose):
    pose = geometry_msgs.msg.Pose()
    pose.position.x = airsim_pose.position.x_val
    pose.position.y = airsim_pose.position.y_val
    pose.position.z = airsim_pose.position.z_val
    pose.orientation.x = airsim_pose.orientation.x_val
    pose.orientation.y = airsim_pose.orientation.y_val
    pose.orientation.z = airsim_pose.orientation.z_val
    pose.orientation.w = airsim_pose.orientation.w_val
    return pose


def randomQuadPose(x_range, y_range, z_range, yaw_range, pitch_range, roll_range):
    x = randomSample(x_range)
    y = randomSample(y_range)
    z = randomSample(z_range)
    yaw = randomSample(yaw_range)
    pitch = randomSample(pitch_range)
    roll = randomSample(roll_range)
    q = transformations.quaternion_from_euler(yaw, pitch, roll, axes='rzyx')
    t_o_b = Vector3r(x,y,z)
    q_o_b = Quaternionr(q[0], q[1], q[2], q[3])
    return Pose(t_o_b, q_o_b), yaw


def randomQuadPose_old(x_range, y_range, z_range, yaw_range, pitch_range, roll_range):
    pose = geometry_msgs.msg.Pose()
    pose.position.x = randomSample(x_range)
    pose.position.y = randomSample(y_range)
    pose.position.z = randomSample(z_range)
    yaw = randomSample(yaw_range)
    pitch = randomSample(pitch_range)
    roll = randomSample(roll_range)
    q = transformations.quaternion_from_euler(yaw, pitch, roll, axes='rzyx')
    pose.orientation = tfQuat2msgQuat(q)
    pose_airsim = rosPose2airsimPose(pose)
    return pose_airsim


def randomGatePose(p_o_b, r_range, cam_fov):
    # create translation of gate
    r = randomSample(r_range)
    alpha = cam_fov/180.0*np.pi/2.0  # alpha is half of fov angle
    theta_range = [-alpha, alpha]
    psi_range = [np.pi/2 -alpha, np.pi/2 + alpha]
    theta = randomSample(theta_range)
    psi = randomSample(psi_range)
    # get relative vector in the base frame
    t_b_g = polarTranslation(r, theta, psi)

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
    point_converted = t.transformPoint('world_frame',point)
    t_o_g = Vector3r(point_converted.point.x, point_converted.point.y, point_converted.point.z)  # now as a world coord vector
    # t_o_g = p_o_b.position + t_b_g

    # create rotation of gate
    # find vector t_b_g in world coordinates
    t_b_g = t_o_g - p_o_b.position
    phi_quad_ref = np.arctan2(t_b_g.y_val, t_b_g.x_val)
    eps = 0  # np.pi/15.0
    phi_rel_range = [-np.pi + eps, 0 - eps]
    phi_rel = randomSample(phi_rel_range)
    phi_gate = phi_quad_ref + phi_rel
    # HACK: to make the red side appear, add 180 to gate pose to be spawned
    # TODO: remove this hack later!!!!!!!!!!!!!!!!!!!!!!!!!!!
    q = transformations.quaternion_from_euler(phi_gate + np.pi, 0, 0, axes='rzyx')
    # q = transformations.quaternion_from_euler(phi_gate, 0, 0, axes='rzyx')
    p_o_g = Pose(t_o_g, Quaternionr(q[0], q[1], q[2], q[3]))
    return p_o_g, r, theta, psi, phi_rel


def randomGatePose_old(x_range, y_range, z_range, yaw_range):
    pose = geometry_msgs.msg.Pose()
    pose.position.x = randomSample(x_range)
    pose.position.y = randomSample(y_range)
    pose.position.z = randomSample(z_range)
    yaw = randomSample(yaw_range)
    q = transformations.quaternion_from_euler(0, 0, yaw)
    pose.orientation = tfQuat2msgQuat(q)
    pose_airsim = rosPose2airsimPose(pose)
    return pose_airsim

def polarTranslation(r, theta, psi):
    # follow math convention for polar coordinates
    # r: radius
    # theta: azimuth (horizontal)
    # psi: vertical
    x = r * np.cos(theta) * np.sin(psi)
    y = r * np.sin(theta) * np.sin(psi)
    z = r * np.cos(psi)
    return Vector3r(x, y, z)

def randomPolarTranslation(r_range, theta_range, psi_range):
    # r: radius
    # theta: azimuth (horizontal)
    # psi: vertical
    radius = randomSample(r_range)
    theta = randomSample(theta_range)
    psi = randomSample(psi_range)
    x = radius * np.sin(theta) * np.cos(psi)
    y = radius * np.sin(theta) * np.sin(psi)
    z = radius * np.cost(theta)
    return Vector3r(x, y, z)


def randomQuat(noise_amp):
    noise_range = [-noise_amp, noise_amp]
    pitch = randomSample(noise_range)
    roll = randomSample(noise_range)
    yaw = randomSample(noise_range)
    quat = to_quaternion(pitch, roll, yaw)  # whatever order, because it's just noise
    quat = normalizeQuat(quat)
    return quat


def randomOffset(value_range):
    offset_x = randomSample(value_range)
    offset_y = randomSample(value_range)
    offset_z = randomSample(value_range)
    return Vector3r(offset_x, offset_y, offset_z)


def getRelativePose(vehicle_pose, gate_pose):
    relative_pose = airsim.Pose()
    relative_pose.position = gate_pose.position - vehicle_pose.position
    # CHANGE THIS LINE HEREHEHREHRHE
    # CHANGEGYTOEMRTKOMREOTKEORMT
    # relative_pose.orientation = vehicle_pose.orientation * gate_pose.orientation * vehicle_pose.orientation.inverse()
    relative_pose.orientation = vehicle_pose.orientation
    return relative_pose


def rotate(vector, quaternion):
    vector = np.array((vector.x_val, vector.y_val, vector.z_val))
    quat_vec = np.array((quaternion.x_val, quaternion.y_val, quaternion.z_val))
    s = quaternion.w_val
    vprime = 2 * np.dot(quat_vec, vector) * quat_vec + (s*s - np.dot(quat_vec, quat_vec)) * vector + 2 * s * np.cross(quat_vec, vector)
    return Vector3r(vprime[0], vprime[1], vprime[2])

