from PoseSampler import *

num_samples = 1000
dataset_path = '/home/rb/data/airsim_datasets/soccer_new_test'
pose_sampler = PoseSampler(num_samples, dataset_path)

# pose_sampler.update_debug()

for idx in range(pose_sampler.num_samples):
    pose_sampler.update()
    if idx % 1000 == 0:
        print('Num samples: {}'.format(idx))
#     # sleep(0.3)   #comment this out once you like your ranges of values
