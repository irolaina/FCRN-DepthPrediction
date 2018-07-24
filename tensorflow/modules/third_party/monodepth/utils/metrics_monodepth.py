import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2
from .evaluation_utils import *
from tqdm import tqdm

def evaluate(args, pred_array, gt_array, args_gt_path):
    # --------------------------------------------- #
    #  Generate Depth Maps for kitti, eigen splits  #
    # --------------------------------------------- #
    # pred_disparities = np.load(args.predicted_disp_path)

    if args.test_split == 'kitti':
        num_samples = 200

        # The FCRN predicts meters instead of disparities, so it's not necessary to convert disps to depth!!!
        gt_disparities = load_gt_disp_kitti(args_gt_path)
        # gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_kitti(gt_disparities, pred_disparities)
        gt_depths = convert_gt_disps_to_depths_kitti(gt_disparities)

        print('\n[Metrics] Generating depth maps...')
        # num_samples = 5  # Only for testing!
        for t_id in tqdm(range(num_samples)):
            # Show the Disparity/Depth ground truths and the corresponding predictions for the evaluation images.
            if False: # TODO: ativar pela flag -u
                print("gt_disp:", gt_disparities[t_id])
                print(np.min(gt_disparities[t_id]), np.max(gt_disparities[t_id]))
                print()
                print("gt_depth:", gt_depths[t_id])
                print(np.min(gt_depths[t_id]), np.max(gt_depths[t_id]))
                print()
                print("pred:", pred_array[t_id])
                print(np.min(pred_array[t_id]), np.max(pred_array[t_id]))
                input()

                plt.figure(100)
                plt.imshow(gt_disparities[t_id])
                plt.title('gt_disp')

                plt.figure(101)
                plt.imshow(gt_depths[t_id].astype(np.uint8))
                plt.title('gt_depth')

                plt.figure(102)
                plt.imshow(pred_array[t_id])
                plt.title('pred')

                plt.draw()
                plt.pause(0.01)

                input("Continue...")

    elif args.test_split == 'eigen':
        num_samples = 697
        test_files = read_text_lines(args.test_file_path)
        gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, args_gt_path)

        num_test = len(im_files)
        gt_depths = []
        pred_depths = []
        print('\n[Metrics] Generating depth maps...')
        # num_samples = 5  # Only for testing!
        for t_id in tqdm(range(num_samples)):
            camera_id = cams[t_id]  # 2 is left, 3 is right
            depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True)
            gt_depths.append(depth.astype(np.float32))

            # The FCRN predicts meters instead of disparities, so it's not necessary to convert disps to depth!!!
            # disp_pred = cv2.resize(pred_disparities[t_id], (im_sizes[t_id][1], im_sizes[t_id][0]), interpolation=cv2.INTER_LINEAR)
            # disp_pred = disp_pred * disp_pred.shape[1]
            #
            # # need to convert from disparity to depth
            # focal_length, baseline = get_focal_length_baseline(gt_calib[t_id], camera_id)
            # depth_pred = (baseline * focal_length) / disp_pred
            # depth_pred[np.isinf(depth_pred)] = 0
            #
            # pred_depths.append(depth_pred)

            # Show the corresponding generated Depth Map from the Stereo Pair.
            if False:
                print(depth)
                input("depth")
                print(pred_array[t_id])
                input("pred")

                plt.figure(100)
                plt.imshow(imageio.imread(im_files[t_id]))
                plt.title('image')

                plt.figure(101)
                plt.imshow(depth.astype(np.uint8))
                plt.title('label')

                plt.figure(102)
                plt.imshow(pred_array[t_id])
                plt.title('pred')

                plt.draw()
                plt.pause(0.01)

    else:
        num_samples = len(pred_array)
        gt_depths = gt_array

    pred_depths = pred_array

    # ----------------- #
    #  Compute Metrics  #
    # ----------------- #
    rms     = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel  = np.zeros(num_samples, np.float32)
    d1_all  = np.zeros(num_samples, np.float32)
    a1      = np.zeros(num_samples, np.float32)
    a2      = np.zeros(num_samples, np.float32)
    a3      = np.zeros(num_samples, np.float32)

    print('\n[Metrics] Computing metrics...')
    # num_samples = 5 # Only for testing!
    for i in tqdm(range(num_samples)):

        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth

        if args.test_split == 'eigen': # TODO: Validar!
            mask = np.logical_and(gt_depth > args.min_depth, gt_depth < args.max_depth)

            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth.shape

                # crop used by Garg ECCV16
                # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
                if args.garg_crop:
                    crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,
                                     0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
                # crop we found by trial and error to reproduce Eigen NIPS14 results
                elif args.eigen_crop:
                    crop = np.array([0.3324324 * gt_height,  0.91351351 * gt_height,
                                     0.0359477 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

        if args.test_split == 'kitti':
            # The FCRN predicts meters instead of disparities, so it's not necessary to convert disps to depth!!!
            # gt_disp = gt_disparities[i]
            # mask = gt_disp > 0
            # pred_disp = pred_disparities_resized[i]
            #
            # disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
            # bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
            # d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

            mask = gt_depth > 0
        else:
            mask = gt_depth > 0

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])
        # abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], gt_depth[mask]) # Only for Testing

        # Show gt/pred Images
        if False: # TODO: Remover?
            plt.figure(102)
            plt.title('pred')
            plt.imshow(pred_depth)

            plt.figure(103)
            plt.imshow(gt_depth)
            plt.title('left')

            plt.draw()
            plt.pause(1)

    # TODO: Implementar
    # Save results on .txt file

    # TODO: adicionar split como informação
    # Display Results
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
