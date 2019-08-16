import numpy as np
from matplotlib import pyplot as plt
import cv2
import time

# open data
data_path = '/home/rb/data/il_datasets/bc_0'
vel_table = np.loadtxt(data_path + '/proc_vel.txt', delimiter=',').astype(np.float32)
with open(data_path + '/proc_images.txt') as f:
    img_table = f.read().splitlines()

# sanity check
if vel_table.shape[0] != len(img_table):
    raise Exception('Number of images ({}) different than number of entries in table ({}): '.format(len(img_table), vel_table.shape[0]))

idx_list = range(vel_table.shape[0])
idx_range = [5000,10000]
plt.plot(idx_list[idx_range[0]:idx_range[1]], vel_table[idx_range[0]:idx_range[1], 0], color='red')
plt.plot(idx_list[idx_range[0]:idx_range[1]], vel_table[idx_range[0]:idx_range[1], 1], color='green')
plt.plot(idx_list[idx_range[0]:idx_range[1]], vel_table[idx_range[0]:idx_range[1], 2], color='blue')
plt.plot(idx_list[idx_range[0]:idx_range[1]], vel_table[idx_range[0]:idx_range[1], 3], color='cyan')
plt.show()
time.sleep(0.5)
plt.close()

vel_scale = 10
yaw_scale = 40
for img_idx in range(10000):
    img = cv2.imread(img_table[img_idx])
    o_x = int(img.shape[0]/2)
    o_y = int(img.shape[1]/2)
    origin = (o_x, o_y)
    pt_vx = (o_x, o_y - int(vel_scale * vel_table[img_idx, 0]))
    pt_vy = (o_x + int(vel_scale * vel_table[img_idx, 1]), o_y)
    cv2.arrowedLine(img, origin, pt_vx, (255, 0, 0), 3)
    cv2.arrowedLine(img, origin, pt_vy, (0, 255, 0), 3)
    cv2.imshow('image', img)
    # time.sleep(0.5)
    cv2.waitKey(20)
