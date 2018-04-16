# ===========
#  Libraries
# ===========
import numpy as np
import tensorflow as tf

# ==================
#  Global Variables
# ==================
TRAINING_L2NORM_BETA = 1e-3
LOG_INITIAL_VALUE = 1


# ===========
#  Functions
# ===========
def np_maskOutInvalidPixels(y, y_):
    condition = y_ <= 0
    idx_i, idx_j = np.where(condition)

    y_masked = np.copy(y)
    for k in range(0, len(idx_i)):
        y_masked[idx_i[k], idx_j[k]] = 0.0  # Predictions with labels equal to zero are set to zero.

    return y_masked


def tf_maskOutInvalidPixels(tf_pred, tf_labels):
    # Identify Pixels to be masked out.
    tf_idx = tf.where(tf_labels > 0)  # Tensor 'idx' of Valid Pixel values (batchID, idx)

    # Mask Out Pixels without depth values
    tf_valid_pred = tf.gather_nd(tf_pred, tf_idx)
    tf_valid_labels = tf.gather_nd(tf_labels, tf_idx)
    tf_valid_log_labels = tf.log(tf_valid_labels, name='log_labels')  # TODO: Precisa daquela constante inicial?

    return tf_valid_pred, tf_valid_labels, tf_valid_log_labels

# ======
#  Loss
# ======
# Attention! The maximum range that the Network can predict depends on the maximum distance recorded in the dataset.
# NYU-Depth v2: ~10 m               / Data range: ]0, ~4000] # TODO: Validar data range values
# Make3D: ~70 m                     / Data range:
# Kitti 2012/2015: ???              / Data range: [0, ~30000] # TODO: Descobrir o range em metros e validar data range values
# KittiRaw Continuous (Vitor): ~80m / Data range: [0, 240]

# -------------------- #
#  Mean Squared Error  #
# -------------------- #
def np_MSE(y, y_):
    numPixels = y_.size

    return np.power(y_ - y, 2) / numPixels  # MSE calculated for each pixel


def tf_MSE(tf_y, tf_y_, valid_pixels=True):
    loss_name = 'MSE'

    # Mask Out
    if valid_pixels:
        tf_y, tf_y_, tf_log_y_ = tf_maskOutInvalidPixels(tf_y, tf_y_)
    else:
        tf_log_y_ = tf.log(tf.cast(tf_y_, tf.float32) + tf.constant(LOG_INITIAL_VALUE, dtype=tf.float32),
                           name='log_labels')  # Just for displaying Image

    # npixels value depends on valid_pixels flag:
    # npixels = (batchSize*height*width) OR npixels = number of valid pixels
    tf_npixels = tf.cast(tf.size(tf_log_y_), tf.float32)

    # Loss
    mse = (tf.reduce_sum(tf.square(tf_log_y_ - tf_y)) / tf_npixels)

    return loss_name, mse


# ------- #
#  BerHu  #
# ------- #
def tf_BerHu(tf_y, tf_y_, valid_pixels=True):
    loss_name = 'BerHu'

    # C Constant Calculation
    tf_log_y_ = tf.log(tf.cast(tf_y_, tf.float32) + tf.constant(LOG_INITIAL_VALUE, dtype=tf.float32),
                       name='log_labels')  # Just for displaying Image
    tf_abs_error = tf.abs(tf.subtract(tf_y, tf_log_y_), name='abs_error')
    tf_c = 0.2 * tf.reduce_max(tf_abs_error)  # Consider All Pixels!

    # Mask Out
    if valid_pixels:
        # Overwrites the 'log(y)' tensor!
        tf_y, tf_y_, tf_log_y_ = tf_maskOutInvalidPixels(tf_y, tf_y_)

        # Overwrites the previous tensor, so now considers only the Valid Pixels!
        tf_abs_error = tf.abs(tf.subtract(tf_y, tf_log_y_), name='abs_error')

    # Loss
    tf_berHu_loss = tf.where(tf_abs_error <= tf_c, tf_abs_error,
                             (tf.square(tf_abs_error) + tf.square(tf_c)) / (2 * tf_c))

    tf_loss = tf.reduce_sum(tf_berHu_loss)

    # Debug
    # c, abs_error, berHu_loss, loss = sess.run([tf_c, tf_abs_error, tf_berHu_loss, tf_loss])
    # print()
    # print(tf_c)
    # print("c:", c)
    # print()
    # print(tf_abs_error)
    # print("abs_error:", abs_error)
    # print(len(abs_error))
    # print()
    # print(tf_berHu_loss)
    # print("berHu_loss:", berHu_loss)
    # print()
    # print(tf_loss)
    # print("loss:", loss)
    # print()
    # input("remover")

    return loss_name, tf_loss


# ------------------------------ #
#  Training Loss - Eigen,Fergus  #
# ------------------------------ #
def gradient_x(img):
    gx = img[:, :, :-1] - img[:, :, 1:]

    # Debug
    # print("img:", img.shape)
    # print("gx:",gx.shape)

    return gx


def gradient_y(img):
    gy = img[:, :-1, :] - img[:, 1:, :]

    # Debug
    # print("img:", img.shape)
    # print("gy:",gy.shape)

    return gy


def tf_L(tf_log_y, tf_log_y_, valid_pixels=True, gamma=0.5):
    loss_name = "Eigen's Log Depth"

    # Calculate Difference and Gradients. Compute over all pixels!
    tf_d = tf_log_y - tf_log_y_
    tf_gx_d = gradient_x(tf_d)
    tf_gy_d = gradient_y(tf_d)

    # Mask Out
    if valid_pixels:
        # Identify Pixels to be masked out.
        tf_idx = tf.where(tf_log_y_ > 0)  # Tensor 'idx' of Valid Pixel values (batchID, idx)

        # Overwrites the 'd', 'gx_d', 'gy_d' tensors, so now considers only the Valid Pixels!
        tf_d = tf.gather_nd(tf_d, tf_idx)
        tf_gx_d = tf.gather_nd(tf_gx_d, tf_idx)
        tf_gy_d = tf.gather_nd(tf_gy_d, tf_idx)

    # Loss
    tf_npixels = tf.cast(tf.size(tf_d), tf.float32)
    mean_term = (tf.reduce_sum(tf.square(tf_d)) / tf_npixels)
    variance_term = ((gamma / tf.square(tf_npixels)) * tf.square(tf.reduce_sum(tf_d)))
    grads_term = (tf.reduce_sum(tf.square(tf_gx_d)) + tf.reduce_sum(tf.square(tf_gy_d))) / tf_npixels

    tf_loss_d = mean_term - variance_term + grads_term

    return loss_name, tf_loss_d


# ------------------ #
#  L2 Normalization  #
# ------------------ #
def getGlobalVars(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


def getTrainableVars(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


def calculateL2norm():
    # Gets All Trainable Variables
    var_list = getTrainableVars('')

    totalSum = 0
    for var in var_list:
        # print(var)
        totalSum += tf.nn.l2_loss(var)

    return TRAINING_L2NORM_BETA * totalSum
