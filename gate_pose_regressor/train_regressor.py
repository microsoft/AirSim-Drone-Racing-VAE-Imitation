import tensorflow as tf
import os
import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))

# import model
models_path = os.path.join(curr_dir, '..', 'racing_models')
sys.path.insert(0, models_path)
import racing_models.dronet

# import utils
models_path = os.path.join(curr_dir, '..', 'racing_utils')
sys.path.insert(0, models_path)
import racing_utils

###########################################

# DEFINE TRAINING META PARAMETERS
data_dir = '/home/rb/data/airsim_datasets/soccer_bright_100k'
output_dir = '/home/rb/data/model_outputs/reg_1'
batch_size = 64
epochs = 10000
img_res = 96

###########################################
# CUSTOM TF FUNCTIONS


@tf.function
def reset_metrics():
    train_loss_rec_gate.reset_states()
    test_loss_rec_gate.reset_states()


@tf.function
def compute_loss(images, labels, predictions):
    recon_loss = tf.losses.mean_squared_error(labels, predictions)
    return recon_loss


@tf.function
def train(images, labels, epoch):
    with tf.GradientTape() as tape:
        predictions = model(images)
        recon_loss = tf.reduce_mean(compute_loss(images, labels, predictions))
        loss = recon_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss_rec_gate.update_state(recon_loss)


@tf.function
def test(images, labels):
    predictions = model(images)
    recon_loss = tf.reduce_mean(compute_loss(images, labels, predictions))
    test_loss_rec_gate.update_state(recon_loss)

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
train_ds, test_ds = racing_utils.dataset_utils.create_dataset_csv(data_dir, batch_size, img_res, num_channels=3)
print('Done with dataset')

# create model
model = racing_models.dronet.Dronet(num_outputs=4, include_top=True)
optimizer = tf.keras.optimizers.Adam(lr=1e-4)

# define metrics
train_loss_rec_gate = tf.keras.metrics.Mean(name='train_loss_rec_gate')
test_loss_rec_gate = tf.keras.metrics.Mean(name='test_loss_rec_gate')
metrics_writer = tf.summary.create_file_writer(output_dir)

# check if output folder exists
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# train
print('Start training ...')
flag = True
for epoch in range(epochs):
    # print('MODE NOW: {}'.format(mode))
    for train_images, train_labels in train_ds:
        train(train_images, train_labels, epoch)
        if flag:
            model.summary()
            flag = False
    for test_images, test_labels in test_ds:
        test(test_images, test_labels)
    # save model
    if epoch % 10 == 0 and epoch > 0:
        print('Saving weights to {}'.format(output_dir))
        model.save_weights(os.path.join(output_dir, "reg_model_{}.ckpt".format(epoch)))

    with metrics_writer.as_default():
        tf.summary.scalar('train_loss_rec_gate', train_loss_rec_gate.result(), step=epoch)
        tf.summary.scalar('test_loss_rec_gate', test_loss_rec_gate.result(), step=epoch)
    print('Epoch {} | L_gate: {} | L_gate: {}'
          .format(epoch, train_loss_rec_gate.result(), test_loss_rec_gate.result()))
    reset_metrics() # reset all the accumulators of metrics

print('bla')