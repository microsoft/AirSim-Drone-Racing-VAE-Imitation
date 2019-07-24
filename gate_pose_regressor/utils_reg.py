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
