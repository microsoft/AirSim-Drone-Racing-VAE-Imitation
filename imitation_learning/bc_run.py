import tensorflow as tf
from bc_utils import create_dataset_txt
from bc_model import ImgRegressor
import argparse
import os

parser = argparse.ArgumentParser()

# parser.add_argument('--path', '-path', help='model file path', default='/home/rb/catkin_ws/src/AutonomousDrivingCookbook/AirSimE2EDeepLearning/output_128_test/regmodel0005.ckpt', type=str)
parser.add_argument('--n_z', '-n_z', help='size of the each one of the parameters [mean,stddev] in the latent space', default=10, type=int)
# parser.add_argument('--data_dir', '-data_dir', help='path to raw data folder', default='/home/rb/data/airsim_datasets/soccer_relative_poses_polar', type=str)
parser.add_argument('--data_dir', '-data_dir', help='path to raw data folder', default='/home/rb/data/airsim_datasets/soccer_new_1k', type=str)
parser.add_argument('--n_gates', '-n_gates', help='number of gates that we are encoding from the image', default=4, type=int)
parser.add_argument('--res', '-res', help='destination resolution for images in the cooked data. if 0, do nothing', default=224, type=int)
parser.add_argument('--num_channels', '-num_channels', help='num of channels in image', default=3, type=int)

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-path', help='model file path', default='/home/rb/catkin_ws/src/AutonomousDrivingCookbook/AirSimE2EDeepLearning/output_reg8/regressor_model_60.ckpt', type=str)
parser.add_argument('--data_dir', '-data_dir', help='path to raw data folder', default='/home/rb/data/il_datasets/il_2', type=str)
parser.add_argument('--output_dir', '-output_dir', help='path to output folder', default='/home/rb/data/il_datasets/il_2/output_bc_0', type=str)
parser.add_argument('--batch_size', '-batch_size', help='number of samples in one minibatch', default=32, type=int)
parser.add_argument('--epochs', '-epochs', help='number of epochs to train the model', default=1000, type=int)
parser.add_argument('--res', '-res', help='destination resolution for images in the cooked data. if 0, do nothing', default=120, type=int)
parser.add_argument('--num_channels', '-num_channels', help='num of channels in image', default=3, type=int)
args = parser.parse_args()

# tf function for prediction
@tf.function
def predict_image(image):
    return model(image)


def interp_vector(a, b, n):
    delta = (b-a)/(n-1)
    list_vecs = []
    for i in range(n):
        new_vec = a+delta*i
        list_vecs.append(new_vec)
    return np.asarray(list_vecs)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Create model
# model = Regressor(args.n_z, args.n_gates, trainable_base_model=True, res=args.res)
model = RegressorDronet(res=args.res)
model.load_weights(args.path)

# Load desired images
files_list = glob.glob(os.path.join(args.data_dir, 'images_test/*.png'))
files_list.sort() # make sure we're reading the images in order later
images_list = []
for file in files_list:
    if args.num_channels == 1:
        im = Image.open(file).resize((args.res, args.res), Image.BILINEAR).convert('L')
        im = np.expand_dims(np.array(im), axis=-1) / 255.0 * 2 - 1.0  # add one more axis and convert to the -1 -> 1 scale
    elif args.num_channels == 3:
        im = Image.open(file).resize((args.res, args.res), Image.BILINEAR)
        im = np.array(im)/255.0*2 - 1.0  # convert to the -1 -> 1 scale
    images_list.append(im)
images_np = np.array(images_list).astype(np.float32)

predictions = predict_image(images_np)
# predictions = predict_image(images_np)

# de-normalization of distances
r_range = [3.0, 10.0]
cam_fov = 90.0  # in degrees -- needs to be a bit smaller than 90 in fact because of cone vs. square
alpha = cam_fov / 180.0 * np.pi / 2.0  # alpha is half of fov angle
theta_range = [-alpha, alpha]
psi_range = [np.pi / 2.0 - alpha, np.pi / 2.0 + alpha]
eps = 0.0  # np.pi / 15.0
phi_rel_range = [-np.pi + eps, 0 - eps]

predictions = predictions.numpy()

# process u and v to create theta and psi
predictions[:, 1] = np.arctan2(predictions[:, 1] * np.tan(alpha), 1.0)
predictions[:, 2] = np.pi/2.0 - np.arctan2(predictions[:, 2] * np.tan(alpha), 1.0)

predictions[:, 0] = (predictions[:, 0] + 1.0)/2.0*(r_range[1]-r_range[0]) + r_range[0]
# predictions[:, 1] = (predictions[:, 1] + 1.0)/2.0*(theta_range[1]-theta_range[0]) + theta_range[0]
# predictions[:, 2] = (predictions[:, 2] + 1.0)/2.0*(psi_range[1]-psi_range[0]) + psi_range[0]
predictions[:, 3] = (predictions[:, 3] + 1.0)/2.0*(phi_rel_range[1]-phi_rel_range[0]) + phi_rel_range[0]

# print('Distance prediction = {}'.format(predictions))

raw_table = np.loadtxt('/home/rb/data/airsim_datasets/soccer_relative_poses_polar/gate_training_data.csv', delimiter=' ')
raw_table.astype(np.float32)

# join predictions with array and print
num_results = 50
indices = np.array([np.arange(num_results)]).transpose()
results = np.concatenate((indices, raw_table[:num_results, :], predictions), axis=1)
# print('Ground truth distance = {}'.format(distances[4]))
print('Img index | Ground-truth values | Predictions: = \n{}'.format(results))

