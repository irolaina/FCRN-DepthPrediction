# ===========
#  Libraries
# ===========
import numpy as np


# ===========
#  Functions
# ===========
# TODO: Metodo abaixo só funciona com o NYUDepth
# FIXME: Acredito que seja necessário adicionar um valor LOG_INITIAL_VALUE, mas nao sei se a metrica permite isso
# TODO: funciona pra todos os datasets?
# Link: https://github.com/iro-cp/FCRN-DepthPrediction/issues/45
def evaluate(pred_array, gt_array):
    # print("Before")
    # print(pred_array.shape, pred_array.dtype)
    # print(gt_array.shape, gt_array.dtype)

    valid_px = False
    if valid_px:
        # Mask Valid Values
        mask = gt_array > 0

        pred = pred_array[mask]
        gt = gt_array[mask]

    else:
        pred = pred_array
        gt = gt_array

    # print("After")
    # print(pred.shape, pred.dtype)
    # print(gt.shape, gt.dtype)

    thr = np.maximum((gt / pred), (pred / gt))
    d1 = (thr < 1.25).mean()
    d2 = (thr < 1.25 ** 2).mean()
    d3 = (thr < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

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
