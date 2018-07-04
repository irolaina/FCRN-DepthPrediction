# Metrics presented by David Eigen, Christian Puhrsch and Rob Fergus in the article "Depth Map Prediction from a Single
# Image using a Multi-Scale Deep Network"
# ===========
#  Libraries
# ===========
import numpy as np


# ===========
#  Functions
# ===========
def evaluate(pred_array, gt_array):
    # Calculates Metrics
    # TODO: Função abaixo pode ser otimizada, uma vez q não é necessario calcular thr = np.maximum... 3 vezes.
    d1 = np_Threshold(pred_array, gt_array, thr=1.25)
    d2 = np_Threshold(pred_array, gt_array, thr=1.25 ** 2)
    d3 = np_Threshold(pred_array, gt_array, thr=1.25 ** 3)

    rmse = np_RMSE_linear(pred_array, gt_array)

    rmse_log = np_RMSE_log(pred_array, gt_array)

    abs_rel = np_AbsRelativeDifference(pred_array, gt_array)

    sq_rel = np_SquaredRelativeDifference(pred_array, gt_array)

    rmse_log_scaleinv = np_RMSE_log_scaleInv(pred_array, gt_array)

    print()
    print("# ----------------- #")
    print("#  Metrics Results  #")
    print("# ----------------- #")
    # # print("thr:", thr)
    # print("d1:", d1)
    # print("d2:", d2)
    # print("d3:", d3)
    # print("rmse:", rmse)
    # print("rmse_log:", rmse_log)
    # print("abs_rel:", abs_rel)
    # print("sq_rel:", sq_rel)
    # print("rmse_log_scaleinv:", rmse_log_scaleinv)
    # # input("metrics")

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms',
                                                                                  'd1_all', 'd1', 'd2', 'd3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel,
                                                                                                  sq_rel,
                                                                                                  rmse,
                                                                                                  rmse_log,
                                                                                                  0.0,
                                                                                                  d1,
                                                                                                  d2,
                                                                                                  d3))


# ------------------- #
#  Mask Valid Pixels  #
# ------------------- #
# TODO: Métrica é aplicada em todos ou apenas nos pixeis válidos? TODOS
def np_maskOutInvalidPixels(y, y_):
    # Index Vectors for Valid Pixels
    nvalids_idx = np.where(y_ > 0)

    # Masking Out Invalid Pixels!
    y = y[nvalids_idx[0], nvalids_idx[1], nvalids_idx[2]]
    y_ = y_[nvalids_idx[0], nvalids_idx[1], nvalids_idx[2]]

    npixels_valid = len(nvalids_idx[0])

    return y, y_, nvalids_idx, npixels_valid


# ----------- #
#  Threshold  #
# ----------- #
def np_Threshold(y, y_, thr):
    # Check if y and y* have the same dimensions
    assert (y.shape == y_.shape), "Houston we've got a problem"

    # Mask out invalid values (values <= 0)!
    y, y_, nvalids_idx, npixels_valid = np_maskOutInvalidPixels(y, y_)

    # Calculates Threshold: % of yi s. t. max(yi/yi*, yi*/yi) = delta < thr
    sigma = np.zeros(npixels_valid, dtype=np.float64)
    for i in range(npixels_valid):
        sigma[i] = max(y[i] / y_[i], y_[i] / y[i])

    value = float(np.sum(sigma < thr)) / float(npixels_valid)

    return value


# ------------------------- #
#  Abs Relative Difference  #
# ------------------------- #
def np_AbsRelativeDifference(y, y_):
    # Check if y and y* have the same dimensions
    assert (y.shape == y_.shape), "Houston we've got a problem"

    npixels_total = np.size(y)  # batchSize*height*width

    # print("Before Mask")
    # print(y.shape)
    # print(y_.shape)
    # print()

    # Mask out invalid values (values <= 0)!
    y, y_, nvalids_idx, npixels_valid = np_maskOutInvalidPixels(y, y_)

    # print("After Mask")
    # print(y.shape)
    # print(y_.shape)
    # print("valid/total: %d/%d" % (npixels_valid, npixels_total))
    # print()

    # Calculate Absolute Relative Difference
    value = sum(abs(y - y_) / y_) / abs(npixels_total)
    # value = sum(abs(y - y_) / y_) / abs(npixels_valid)

    return value


# ----------------------------- #
#  Squared Relative Difference  #
# ----------------------------- #
def np_SquaredRelativeDifference(y, y_):
    # Check if y and y* have the same dimensions
    assert (y.shape == y_.shape), "Houston we've got a problem"

    npixels_total = np.size(y)  # batchSize*height*width

    # Mask out invalid values (values <= 0)!
    y, y_, nvalids_idx, npixels_valid = np_maskOutInvalidPixels(y, y_)

    # Calculate Absolute Relative Difference
    value = sum((abs(y - y_) ** 2 / y_) / abs(npixels_total))
    # value = sum(pow((abs(y - y_) / y_), 2) / abs(npixels_valid))

    return value


# -------------- #
#  RMSE(linear)  #
# -------------- #
def np_RMSE_linear(y, y_):
    # Check if y and y* have the same dimensions
    assert (y.shape == y_.shape), "Houston we've got a problem"

    npixels_total = np.size(y)  # batchSize*height*width

    # Mask out invalid values (values <= 0)!
    y, y_, nvalids_idx, npixels_valid = np_maskOutInvalidPixels(y, y_)

    # Calculate Absolute Relative Difference
    value = np.sqrt(sum(pow(abs(y - y_), 2)) / abs(npixels_total))
    # value = np.sqrt(sum(pow(abs(y - y_), 2)) / abs(npixels_valid))

    return value


# ----------- #
#  RMSE(log)  #
# ----------- #
def np_RMSE_log(y, y_):
    # Check if y and y* have the same dimensions
    assert (y.shape == y_.shape), "Houston we've got a problem"

    npixels_total = np.size(y)  # batchSize*height*width

    # Mask out invalid values (values <= 0)!
    y, y_, nvalids_idx, npixels_valid = np_maskOutInvalidPixels(y, y_)

    # Calculate Absolute Relative Difference
    value = np.sqrt(sum(pow(abs(np.log(y) - np.log(y_)), 2)) / abs(npixels_total))
    # value = np.sqrt(sum(pow(abs(np.log(y) - np.log(y_)), 2)) / abs(npixels_valid))

    return value


# ---------------------------- #
#  RMSE(log, scale-invariant)  #
# ---------------------------- #
def np_RMSE_log_scaleInv(y, y_):
    # Check if y and y* have the same dimensions
    assert (y.shape == y_.shape), "Houston we've got a problem"

    npixels_total = np.size(y)  # batchSize*height*width

    # Mask out invalid values (values <= 0)!
    y, y_, nvalids_idx, npixels_valid = np_maskOutInvalidPixels(y, y_)

    # Calculate Absolute Relative Difference
    alfa = sum(np.log(y_) - np.log(y)) / npixels_total
    value = sum(pow(np.log(y) - np.log(y_) + alfa, 2)) / (2 * npixels_total)

    # alfa = sum(np.log(y_) - np.log(y)) / npixels_valid
    # value = sum(pow(np.log(y) - np.log(y_) + alfa, 2)) / npixels_valid

    # Additional computation way
    # d = np.log(y) - np.log(y_)
    # value2 = sum(pow(d,2))/npixels_valid - pow(sum(d),2)/pow(npixels_valid,2)

    return value
