# ===========
#  Libraries
# ===========
import numpy as np
import tensorflow as tf

# ==================
#  Global Variables
# ==================
TRAINING_L2NORM_BETA = 1e-3


# ===========
#  Functions
# ===========
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


def np_maskOutInvalidPixels(y, y_):
    condition = y_ <= 0
    idx_i, idx_j = np.where(condition)

    y_masked = np.copy(y)
    for k in range(0, len(idx_i)):
        y_masked[idx_i[k], idx_j[k]] = 0.0  # Predictions with labels equal to zero are set to zero.

    return y_masked


def tf_maskOutInvalidPixels(tf_y, tf_y_):
    # Values Range
    # NyuDepth - ]0, ~4000]
    # Kitti 2012/2015 - [0, ~30000]
    # Kittiraw Continuous (Vitor) - [0, 255]

    # Variables
    tf_idx = tf.where(tf_y_ > 0)  # Tensor 'idx' of Valid Pixel values (batchID, idx)
    tf_valid_y = tf.gather_nd(tf_y, tf_idx)
    tf_valid_y_ = tf.gather_nd(tf_y_, tf_idx)
    tf_npixels_valid = tf.cast(tf.shape(tf_valid_y_), tf.float32)

    return tf_valid_y, tf_valid_y_, tf_npixels_valid


# -------------------- #
#  Mean Squared Error  #
# -------------------- #
def np_MSE(y, y_):
    numPixels = y_.size

    return np.power(y_ - y, 2) / numPixels  # MSE calculated for each pixel


# TODO: Excluir?
def tf_MSE_monodeep(tf_valid_y, tf_valid_log_y_):
    loss_name = 'MSE'

    tf_npixels_valid = tf.cast(tf.size(tf_valid_log_y_), tf.float32)  # (batchSize*height*width)

    return loss_name, (tf.reduce_sum(tf.pow(tf_valid_log_y_ - tf_valid_y, 2)) / tf_npixels_valid)


def tf_MSE(tf_y, tf_log_y_):
    tf_y = tf.squeeze(tf_y, axis=3)

    loss_name = 'MSE'

    tf_npixels_valid = tf.cast(tf.size(tf_log_y_), tf.float32)  # (batchSize*height*width)

    return loss_name, (tf.reduce_sum(tf.pow(tf_log_y_ - tf_y, 2)) / tf_npixels_valid)


# ------- #
#  BerHu  #
# ------- #
# TODO: Implemente BerHu Loss function
def np_BerHu():
    pass


# TODO: Implemente BerHu Loss function
def tf_BerHu():
    pass


# ------------------------------ #
#  Training Loss - Eigen,Fergus  #
# ------------------------------ #
def tf_L(tf_log_y, tf_log_y_, tf_idx, gamma=0.5):
    loss_name = "Eigen's Log Depth"

    # Calculate Difference and Gradients
    tf_d = tf_log_y - tf_log_y_
    tf_gx_d = gradient_x(tf_d)
    tf_gy_d = gradient_y(tf_d)

    # Mask Out
    tf_valid_d = tf.gather_nd(tf_d, tf_idx)
    tf_valid_gx_d = tf.gather_nd(tf_gx_d, tf_idx)
    tf_valid_gy_d = tf.gather_nd(tf_gy_d, tf_idx)

    # Loss
    tf_valid_npixels = tf.cast(tf.size(tf_valid_d), tf.float32)
    mean_term = (tf.reduce_sum(tf.pow(tf_valid_d, 2)) / tf_valid_npixels)
    variance_term = ((gamma / tf.pow(tf_valid_npixels, 2)) * tf.pow(tf.reduce_sum(tf_valid_d), 2))
    grads_term = (tf.reduce_sum(tf.pow(tf_valid_gx_d, 2)) + tf.reduce_sum(tf.pow(tf_valid_gy_d, 2))) / tf_valid_npixels

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
    coarse_vars = getTrainableVars("c_")
    fine_vars = getTrainableVars("f_")

    totalSum = 0
    for i, val in enumerate(coarse_vars):
        totalSum += tf.nn.l2_loss(val)
        # print (i, val)

    for i, val in enumerate(fine_vars):
        totalSum += tf.nn.l2_loss(val)
        # print (i, val)

    # Debug
    # print(coarse_vars)
    # print(fine_vars)
    # print("totalSum: ", totalSum)

    return TRAINING_L2NORM_BETA * totalSum
