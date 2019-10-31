import time
from PoseSampler import *

num_samples = 100
dataset_path = '/home/rb/all_files/airsim_datasets/soccer_test'

# check if output folder exists
if not os.path.isdir(dataset_path):
    os.makedirs(dataset_path)
    img_dir = os.path.join(dataset_path, 'images')
    os.makedirs(img_dir)
else:
    print('Error: path already exists')

pose_sampler = PoseSampler(num_samples, dataset_path)

for idx in range(pose_sampler.num_samples):
    pose_sampler.update()
    if idx % 1000 == 0:
        print('Num samples: {}'.format(idx))
    # time.sleep(0.3)   #comment this out once you like your ranges of values
