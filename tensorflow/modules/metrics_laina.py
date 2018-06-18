# ===========
#  Libraries
# ===========
import numpy as np

# ===========
#  Functions
# ===========
# TODO: Validar, usar o arquivo error_metrics.m da Iro Laina
# Link: https://github.com/iro-cp/FCRN-DepthPrediction/issues/45
def evaluate(pred_array, gt_array):
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
    rmse_log = (np.log10(gt) - np.log10(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    # mask_aux = np.where(np.log10(pred) < 0)
    # mask_aux2 = np.where(np.isinf(np.log10(pred)))
    # pred2 = pred[mask_aux]
    # pred3 = pred[mask_aux2]
    # print(pred2)
    # print(pred3)
    # print(np.log10(pred2))
    # print(np.log10(pred3))
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
