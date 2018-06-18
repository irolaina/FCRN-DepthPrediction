# Metrics presented by David Eigen, Christian Puhrsch and Rob Fergus in the article "Depth Map Prediction from a Single
# Image using a Multi-Scale Deep Network"
# ===========
#  Libraries
# ===========
import numpy as np

# ===========
#  Functions
# ===========
# TODO: Validar, usar o arquivo error_metrics.m da Iro Laina
# Link: https://github.com/iro-cp/FCRN-DepthPrediction/issues/45
def evaluateTestSetLaina(pred_array, gt_array):
    # print((gt_array / pred_array))
    # print((pred_array / gt_array))
    # thr = np.maximum((gt_array / pred_array), (pred_array / gt_array))

    # pred_aux, gt_aux = [], []
    # for i in range(len(pred_array)):
    #     pred = pred_array[i]
    #     gt = gt_array[i]
    #
    #     mask = np.where(gt > 0)
    #
    #     pred = pred[mask]
    #     gt = gt[mask]
    #
    #     pred_aux.append(pred)
    #     gt_aux.append(gt)
    #
    #     print(pred.shape)
    #     print(gt.shape)
    #
    # pred = np.array(pred_aux)
    # gt = np.array(gt_aux)
    #
    # print(pred.shape, pred.dtype)
    # print(gt.shape, gt.dtype)
    # input("aki")

    # ------------------- #
    #  Mask Valid Pixels  #
    # ------------------- #
    # TODO: Metodo abaixo só funciona com o NYUDepth
    valid_px = False

    print("Before")
    print(pred_array.shape, pred_array.dtype)
    print(gt_array.shape, gt_array.dtype)

    if valid_px:
        # Mask Valid Values
        mask = np.where(gt_array > 0)  # TODO: funciona pra todos os datasets?

        # print(mask)
        # print(len(mask))

        pred = pred_array[mask]
        gt = gt_array[mask]

    else:
        pred = pred_array
        gt = gt_array

    print("After")
    print(pred.shape, pred.dtype)
    print(gt.shape, gt.dtype)

    # ----------- #
    #  Threshold  #
    # ----------- #
    thr = np.maximum((gt / pred), (pred / gt))
    d1 = (thr < 1.25).mean()
    d2 = (thr < 1.25 ** 2).mean()
    d3 = (thr < 1.25 ** 3).mean()

    # -------------- #
    #  RMSE(linear)  #
    # -------------- #
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    # ----------- #
    #  RMSE(log)  #
    # ----------- #
    # TODO: Devo usar log ou log10?
    # FIXME: Acredito que seja necessário adicionar um valor LOG_INITIAL_VALUE, mas nao sei se a metrica permite isso
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    # mask_aux = np.where(np.log(pred) < 0)
    # mask_aux2 = np.where(np.isinf(np.log(pred)))
    # pred2 = pred[mask_aux]
    # pred3 = pred[mask_aux2]
    # print(pred2)
    # print(pred3)
    # print(np.log(pred2))
    # print(np.log(pred3))
    # input("oi")

    # ------------------------- #
    #  Abs Relative Difference  #
    # ------------------------- #
    abs_rel = np.mean(np.abs(gt - pred) / gt)

    # ----------------------------- #
    #  Squared Relative Difference  #
    # ----------------------------- #
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    print()
    print("# ----------------- #")
    print("#  Metrics Results  #")
    print("# ----------------- #")
    # print("thr:", thr)
    print("d1:", d1)
    print("d2:", d2)
    print("d3:", d3)
    print("rmse:", rmse)
    print("rmse_log:", rmse_log)
    print("abs_rel:", abs_rel)
    print("sq_rel:", sq_rel)
    # input("metrics")


def evaluateTesting(fine, labels):
    print("[Network/Testing] Calculating Metrics based on Testing Predictions...")
    print("Input")
    print("predFine:", fine.shape)
    print("labels:", labels.shape)
    print()

    # Calculates Metrics
    print("# ----------------- #")
    print("#  Metrics Results  #")
    print("# ----------------- #")
    print("Threshold sig < 1.25:", np_Threshold(fine, labels, thr=1.25))
    print("Threshold sig < 1.25^2:", np_Threshold(fine, labels, thr=pow(1.25, 2)))
    print("Threshold sig < 1.25^3:", np_Threshold(fine, labels, thr=pow(1.25, 3)))
    print("RMSE(linear):", np_RMSE_linear(fine, labels))
    print("RMSE(log):", np_RMSE_log(fine, labels))
    print("AbsRelativeDifference:", np_AbsRelativeDifference(fine, labels))
    print("SqrRelativeDifference:", np_SquaredRelativeDifference(fine, labels))
    print()
    print("RMSE(log, scale inv.):", np_RMSE_log_scaleInv(fine, labels))


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
