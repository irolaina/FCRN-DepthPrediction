import numpy as np

def compute_errors(pred, gt):
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

    return abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3

def evaluate(pred_array, gt_array):
    num_samples = len(pred_array)
    pred_depths, gt_depths = pred_array, gt_array
    min_depth, max_depth = 1e-3, 80

    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1_all = np.zeros(num_samples, np.float32)
    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)

    for i in range(num_samples):
        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth

        # if split == 'eigen':
        #     mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
        #
        #     if garg_crop or eigen_crop:
        #         gt_height, gt_width = gt_depth.shape
        #
        #         # crop used by Garg ECCV16
        #         # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
        #         if garg_crop:
        #             crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
        #                              0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
        #         # crop we found by trial and error to reproduce Eigen NIPS14 results
        #         elif eigen_crop:
        #             crop = np.array([0.3324324 * gt_height, 0.91351351 * gt_height,
        #                              0.0359477 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
        #
        #         crop_mask = np.zeros(mask.shape)
        #         crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        #         mask = np.logical_and(mask, crop_mask)
        #
        # if split == 'kitti':
        #     gt_disp = gt_disparities[i]
        #     mask = gt_disp > 0
        #     pred_disp = pred_disparities_resized[i]
        #
        #     disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
        #     bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
        #     d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

        mask = gt_depth > 0

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(pred_depth[mask], gt_depth[mask])

    print()
    print("# ----------------- #")
    print("#  Metrics Results  #")
    print("# ----------------- #")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms',
                                                                                  'd1_all', 'd1', 'd2', 'd3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(),
                                                                                                  sq_rel.mean(),
                                                                                                  rms.mean(),
                                                                                                  log_rms.mean(),
                                                                                                  d1_all.mean(),
                                                                                                  a1.mean(), a2.mean(),
                                                                                                  a3.mean()))
