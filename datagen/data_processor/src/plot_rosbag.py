import rosbag
import glob
import os
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
from geometry_msgs.msg import TwistStamped
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# files_list = ['/home/rb/data/log/most_2019-09-02-16-37-41.bag']
print('Going to read file list')
data_dir = '/media/rb/hd_2tb/data/log_3'
# files_list = glob.glob(os.path.join(data_dir, '*.bag'))
files_list = ['/media/rb/hd_2tb/data/log_3/most_2019-09-10-19-20-50.bag']
path_save_images = '/home/rb/data/real_life/real_life_run'

folder_idx = 0
vel_cmds = np.zeros((50000, 4))
for file_name in files_list:
    print('Starting to process bag:' + file_name)
    bag = rosbag.Bag(file_name, 'r')
    vel_idx = 0
    for topic, msg, t in bag.read_messages(topics='/vel_cmd'):
        # if t.to_sec() > 1568157726 and t.to_sec() < 1568157755:
        if t.to_sec() > 1568157726 and t.to_sec() < 1568157745:
            if vel_idx % 500 == 0:
                print('Processed {} vel'.format(vel_idx))
            vel_cmds[vel_idx, 0] = msg.twist.linear.x
            vel_cmds[vel_idx, 1] = msg.twist.linear.y
            vel_cmds[vel_idx, 2] = msg.twist.linear.z
            vel_cmds[vel_idx, 3] = msg.twist.angular.z
            vel_idx = vel_idx + 1
    folder_idx = folder_idx + 1
    bag.close()

# fig, axs = plt.subplots(1, 4, tight_layout=True)
# min_num = 0
# max_num = vel_idx
# axs[0].plot(np.arange(max_num-min_num), vel_cmds[min_num:max_num, 0], 'b-', label='r')
# axs[1].plot(np.arange(max_num-min_num), vel_cmds[min_num:max_num, 1], 'b-', label=r'$\theta$')
# axs[2].plot(np.arange(max_num-min_num), vel_cmds[min_num:max_num, 2], 'b-', label=r'$\phi$')
# axs[3].plot(np.arange(max_num-min_num), vel_cmds[min_num:max_num, 3], 'b-', label=r'$\psi$')

# fig, axs = plt.subplots(1, 1, tight_layout=True)
# min_num = 0
# max_num = vel_idx
# axs.plot(np.arange(vel_idx).astype(np.float32)/vel_idx*19, vel_cmds[min_num:max_num, 0], 'r-', label=r'$V_x$')
# axs.plot(np.arange(vel_idx).astype(np.float32)/vel_idx*19, vel_cmds[min_num:max_num, 1], 'g-', label=r'$V_y$')
# axs.plot(np.arange(vel_idx).astype(np.float32)/vel_idx*19, vel_cmds[min_num:max_num, 2], 'b-', label=r'$V_z$')
# axs.plot(np.arange(vel_idx).astype(np.float32)/vel_idx*19, vel_cmds[min_num:max_num, 3], 'c-', label=r'$V_{\psi}$')
#
# axs.set_title('Velocities commanded to UAV in time')
# axs.set_ylabel('Velocity [m/s] or [rad/s]')
# axs.set_xlabel('Time [seconds]')
# axs.legend()
# axs.grid()

min_num = 0
max_num = vel_idx
sns.set(style="whitegrid")
data = pd.DataFrame(vel_cmds[min_num:max_num], np.arange(vel_idx).astype(np.float32)/vel_idx*19, columns=[r'$V_x$',r'$V_y$',r'$V_z$',r'$V_{\psi}$'])
fig = sns.lineplot(data=data, linewidth=2.5, dashes=False)
fig.set_title('Velocities commanded to UAV in time')
fig.set_ylabel('Velocity [m/s] or [rad/s]')
fig.set_xlabel('Time [seconds]')

# plt.pyplot.show()

print('bla')