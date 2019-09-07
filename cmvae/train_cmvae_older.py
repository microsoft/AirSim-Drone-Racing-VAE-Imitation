import tensorflow as tf
import os
import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))

# import model
models_path = os.path.join(curr_dir, '..', 'racing_models')
sys.path.insert(0, models_path)
import racing_models.cmvae

# import utils
models_path = os.path.join(curr_dir, '..', 'racing_utils')
sys.path.insert(0, models_path)
import racing_utils

###########################################

# DEFINE TRAINING META PARAMETERS
data_dir = '/home/rb/data/airsim_datasets/soccer_small_50k'
output_dir = '/home/rb/data/model_outputs/cmvae_d_0'
batch_size = 32
epochs = 10000
n_z = 10
img_res = 64
max_size = None  # default is None
learning_rate = 1e-4

###########################################
# CUSTOM TF FUNCTIONS


@tf.function
def calc_weighted_loss_img(img_recon, images_np):
    flat_pred = tf.reshape(img_recon, [-1])
    flat_gt = tf.reshape(images_np, [-1])
    error_sq = tf.math.squared_difference(flat_gt, flat_pred)
    softmax_weights = tf.math.exp(error_sq) / tf.reduce_sum(tf.math.exp(error_sq))
    weighted_error_sq = error_sq * softmax_weights
    loss = tf.reduce_sum(weighted_error_sq)
    return loss


def reset_metrics():
    train_loss_rec_img.reset_states()
    train_loss_rec_gate.reset_states()
    train_loss_kl.reset_states()
    test_loss_rec_img.reset_states()
    test_loss_rec_gate.reset_states()
    test_loss_kl.reset_states()


@tf.function
def regulate_weights(epoch):
    # for beta
    if epoch < 10.0:
        beta = 8.0
    else:
        beta = 8.0
    # t = 10
    # beta_min = 0.0  #0.000001
    # beta_max = 1.0  #0.0001
    # if epoch < t:
    #     # beta = beta_min + epoch/t*(beta_max-beta_min)
    #     beta = beta_max * 0.95**(t-epoch)  # ranges from 0.00592052922 to 0.95
    # else:
    #     beta = beta_max
    # for w_img
    if epoch < 100:
        w_img = 1.0
    else:
        w_img = 1.0
    # for w_gate
    if epoch < 100:
        w_gate = 1.0
    else:
        w_gate = 1.0
    return beta, w_img, w_gate


@tf.function
def compute_loss_unsupervised(img_gt, gate_gt, img_recon, gate_recon, means, stddev, mode):
    # copute reconstruction loss
    if mode == 0:
        img_loss = tf.losses.mean_squared_error(img_gt, img_recon)
        # img_loss = tf.losses.mean_absolute_error(img_gt, img_recon)
        gate_loss = tf.losses.mean_squared_error(gate_gt, gate_recon)
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum((1 + stddev - tf.math.pow(means, 2) - tf.math.exp(stddev)), axis=1))
    # elif mode == 1:
    #     # labels = tf.reshape(labels, predictions.shape)
    #     # recon_loss = tf.losses.mean_squared_error(labels, predictions)
    #     # recon_loss = loss_object(labels, predictions)
    # print('Predictions: {}'.format(predictions))
    # print('Labels: {}'.format(labels))
    # print('Lrec: {}'.format(recon_loss))
    # copute KL loss: D_KL(Q(z|X,y) || P(z|X))
    return img_loss, gate_loss, kl_loss


@tf.function
def train(img_gt, gate_gt, epoch, mode):
    # freeze the non-utilized weights
    # if mode == 0:
    #     model.q_img.trainable = True
    #     model.p_img.trainable = True
    #     model.p_gate.trainable = True
    # elif mode == 1:
    #     model.q_img.trainable = True
    #     model.p_img.trainable = True
    #     model.p_gate.trainable = False
    # elif mode == 2:
    #     model.q_img.trainable = True
    #     model.p_img.trainable = False
    #     model.p_gate.trainable = True
    with tf.GradientTape() as tape:
        img_recon, gate_recon, means, stddev, z = model(img_gt, mode)
        img_loss, gate_loss, kl_loss = compute_loss_unsupervised(img_gt, gate_gt, img_recon, gate_recon, means, stddev, mode)
        img_loss = tf.reduce_mean(img_loss)
        gate_loss = tf.reduce_mean(gate_loss)
        beta, w_img, w_gate = regulate_weights(epoch)
        # weighted_loss_img = calc_weighted_loss_img(img_recon, img_gt)
        if mode == 0:
            total_loss = w_img*img_loss + w_gate*gate_loss + beta*kl_loss
            # total_loss = w_img * img_loss + beta * kl_loss
            # total_loss = weighted_loss_img + gate_loss + beta * kl_loss
            # total_loss = img_loss
            train_loss_rec_img.update_state(img_loss)
            train_loss_rec_gate.update_state(gate_loss)
            train_loss_kl.update_state(kl_loss)
        # TODO: later create structure for other training modes -- for now just training everything together
        # elif mode==1:
        #     total_loss = img_loss + beta*kl_loss
        #     train_kl_loss_m1(kl_loss)
        # elif mode==2:
        #     total_loss = gate_loss + beta*kl_loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


@tf.function
def test(img_gt, gate_gt, mode):
    img_recon, gate_recon, means, stddev, z = model(img_gt, mode)
    img_loss, gate_loss, kl_loss = compute_loss_unsupervised(img_gt, gate_gt, img_recon, gate_recon, means, stddev, mode)
    img_loss = tf.reduce_mean(img_loss)
    gate_loss = tf.reduce_mean(gate_loss)
    if mode == 0:
        test_loss_rec_img.update_state(img_loss)
        test_loss_rec_gate.update_state(gate_loss)
        test_loss_kl.update_state(kl_loss)

###########################################


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# load dataset
print('Starting dataset')
train_ds, test_ds = racing_utils.dataset_utils.create_dataset_csv(data_dir, batch_size, img_res, max_size=max_size)
print('Done with dataset')

# create model
# model = racing_models.cmvae.Cmvae(n_z=n_z, gate_dim=4, res=img_res, trainable_model=True)
model = racing_models.cmvae.CmvaeDirect(n_z=n_z, gate_dim=4, res=img_res, trainable_model=True)
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

# define metrics
train_loss_rec_img = tf.keras.metrics.Mean(name='train_loss_rec_img')
train_loss_rec_gate = tf.keras.metrics.Mean(name='train_loss_rec_gate')
train_loss_kl = tf.keras.metrics.Mean(name='train_loss_kl')
test_loss_rec_img = tf.keras.metrics.Mean(name='test_loss_rec_img')
test_loss_rec_gate = tf.keras.metrics.Mean(name='test_loss_rec_gate')
test_loss_kl = tf.keras.metrics.Mean(name='test_loss_kl')
metrics_writer = tf.summary.create_file_writer(output_dir)

# check if output folder exists
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# train
print('Start training ...')
mode = 0
flag = True
for epoch in range(epochs):
    # print('MODE NOW: {}'.format(mode))
    for train_images, train_labels in train_ds:
        train(train_images, train_labels, epoch, mode)
        if flag:
            model.summary()
            flag = False
    for test_images, test_labels in test_ds:
        test(test_images, test_labels, mode)
    # save model
    if epoch % 5 == 0 and epoch > 0:
        print('Saving weights to {}'.format(output_dir))
        model.save_weights(os.path.join(output_dir, "cmvae_model_{}.ckpt".format(epoch)))

    if mode == 0:
        with metrics_writer.as_default():
            tf.summary.scalar('train_loss_rec_img', train_loss_rec_img.result(), step=epoch)
            tf.summary.scalar('train_loss_rec_gate', train_loss_rec_gate.result(), step=epoch)
            tf.summary.scalar('train_loss_kl', train_loss_kl.result(), step=epoch)
            tf.summary.scalar('test_loss_rec_img', test_loss_rec_img.result(), step=epoch)
            tf.summary.scalar('test_loss_rec_gate', test_loss_rec_gate.result(), step=epoch)
            tf.summary.scalar('test_loss_kl', test_loss_kl.result(), step=epoch)
        print('Epoch {} | TRAIN: L_img: {}, L_gate: {}, L_kl: {}, L_tot: {} | TEST: L_img: {}, L_gate: {}, L_kl: {}, L_tot: {}'
              .format(epoch, train_loss_rec_img.result(), train_loss_rec_gate.result(), train_loss_kl.result(),
                      train_loss_rec_img.result()+train_loss_rec_gate.result()+train_loss_kl.result(),
                      test_loss_rec_img.result(), test_loss_rec_gate.result(), test_loss_kl.result(),
                      test_loss_rec_img.result() + test_loss_rec_gate.result() + test_loss_kl.result()
                      ))
        reset_metrics() # reset all the accumulators of metrics

print('bla')