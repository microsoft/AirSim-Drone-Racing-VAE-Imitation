import rosbag
import glob
import os
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2

# files_list = ['/home/rb/data/log/most_2019-09-02-16-37-41.bag']
print('Going to read file list')
data_dir = '/home/rb/data/log_2'
files_list = glob.glob(os.path.join(data_dir, '*.bag'))
path_save_images = '/home/rb/data/real_life'

bridge = CvBridge()
img_res = 64

folder_idx = 0
for file_name in files_list:
    print('Starting to process bag:' + file_name)
    bag = rosbag.Bag(file_name, 'r')

    base_save_path = os.path.join(path_save_images, 'bag_'+str(folder_idx))
    path_images = os.path.join(base_save_path, 'images')
    if not os.path.exists(base_save_path):
        os.mkdir(base_save_path)
        os.mkdir(path_images)
        print('Directory {} created'.format(base_save_path))
    else:
        print('Directory {} already exists'.format(base_save_path))
        exit()
    img_idx = 0

    for topic, msg, t in bag.read_messages(topics='/uav1/camera/color_Image'):
        if img_idx % 500 == 0:
            print('Processed {} images'.format(img_idx))
        img_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8").astype(np.uint8)
        img_bgr = img_bgr[0:240, 40:280]
        img_resized = cv2.resize(img_bgr, (img_res, img_res))
        if img_idx > 2000:
            cv2.imwrite(os.path.join(path_images, str(img_idx).zfill(6) + '.png'), img_resized)
        img_idx = img_idx + 1
    folder_idx = folder_idx + 1
    bag.close()
