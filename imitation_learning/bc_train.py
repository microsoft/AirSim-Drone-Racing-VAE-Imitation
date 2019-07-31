import tensorflow as tf
from bc_utils import create_dataset_txt
from bc_model import ImgRegressor
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', '-data_dir', help='path to raw data folder', default='/home/rb/data/il_datasets/il_3', type=str)
parser.add_argument('--output_dir', '-output_dir', help='path to output folder', default='/home/rb/data/il_datasets/il_2/output_bc_4', type=str)
parser.add_argument('--batch_size', '-batch_size', help='number of samples in one minibatch', default=32, type=int)
parser.add_argument('--epochs', '-epochs', help='number of epochs to train the model', default=1000, type=int)
parser.add_argument('--res', '-res', help='destination resolution for images in the cooked data. if 0, do nothing', default=120, type=int)
parser.add_argument('--num_channels', '-num_channels', help='num of channels in image', default=3, type=int)
args = parser.parse_args()

###########################################
# CUSTOM FUNCTIONS

@tf.function
def reset_metrics():
    train_loss_m1.reset_states()
    test_loss_m1.reset_states()


@tf.function
def compute_loss(images, labels, predictions):
    # labels = tf.reshape(labels, predictions.shape)
    recon_loss = tf.losses.mean_squared_error(labels, predictions)
    # print('Predictions: {}'.format(predictions))
    # print('Labels: {}'.format(labels))
    # print('Lrec: {}'.format(recon_loss))
    # copute KL loss: D_KL(Q(z|X,y) || P(z|X))
    # kl_loss = -0.5*tf.reduce_mean(tf.reduce_sum((1+stddev-tf.math.pow(means, 2)-tf.math.exp(stddev)), axis=1))
    return recon_loss


# tf function to train
@tf.function
def train(images, labels, epoch):
    with tf.GradientTape() as tape:
        predictions = model(images)
        # print("Z: {}".format(z))
        # print("X almost final: {}".format(x_almost_final))
        recon_loss = tf.reduce_mean(compute_loss(images, labels, predictions))
        # # KL weight (to be used by total loss and by annealing scheduler)
        # self.kl_weight = K.variable(self.hps['kl_weight_start'], name='kl_weight')
        # kl_weight = self.kl_weight
        loss = recon_loss
    # print('Mode {} | Loss to backprop: {}'.format(mode, recon_loss))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss_m1(recon_loss)


@tf.function
def test(images, labels):
    predictions = model(images)
    recon_loss = tf.reduce_mean(compute_loss(images, labels, predictions))
    test_loss_m1(recon_loss)

###########################################

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# load dataset
print('Starting dataset')
train_ds, test_ds = create_dataset_txt(args.data_dir, args.batch_size, args.res, args.num_channels)
print('Done with dataset')

# create model
# model = Regressor(args.n_z, args.n_gates, trainable_base_model=True, res=args.res)
model = ImgRegressor(res=args.res)
optimizer = tf.keras.optimizers.Adam(lr=1e-4)

# define metrics
train_loss_m1 = tf.keras.metrics.Mean(name='train_loss_m1')
test_loss_m1 = tf.keras.metrics.Mean(name='test_loss_m1')
metrics_writer = tf.summary.create_file_writer(args.output_dir)

# check if output folder exists
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

# train
print('Start training ...')
flag = True
for epoch in range(args.epochs):
    for images, labels in train_ds:
        train(images, labels, epoch)
        if flag:
            model.summary()
            flag = False
    for test_images, test_labels in test_ds:
        test(test_images, test_labels)
    if epoch % 20 == 0 and epoch > 0:
        print('Saving weights to {}'.format(args.output_dir))
        model.save_weights(os.path.join(args.output_dir, "bc_model_{}.ckpt".format(epoch)))
    with metrics_writer.as_default():
        tf.summary.scalar('Train loss m1', train_loss_m1.result(), step=epoch)
        tf.summary.scalar('Test loss m1', test_loss_m1.result(), step=epoch)
    print('Epoch {}, L_train: {}, L_test: {}'.format(epoch, train_loss_m1.result(), test_loss_m1.result()))
    reset_metrics()  # reset all the accumulators of metrics

