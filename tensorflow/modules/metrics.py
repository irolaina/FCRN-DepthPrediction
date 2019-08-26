import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deprecated import deprecated
from tqdm import tqdm

from modules.args import args
from modules.third_party.monodepth.utils.evaluation_utils import compute_errors
from modules.utils import settings
from modules.third_party.monodepth.utils.evaluation_utils import load_gt_disp_kitti, convert_gt_disps_to_depths_kitti, \
    read_text_lines, read_file_data, \
    generate_depth_map


# ===========
#  Functions
# ===========
def generate_depth_maps_kitti_split(pred_array, args_gt_path):
    num_test_images = 200

    # The FCRN predicts meters instead of disparities, so it's not necessary to convert disps to depth!!!
    gt_disparities = load_gt_disp_kitti(args_gt_path)

    print('\n[Metrics] Generating depth maps...')
    gt_depths = convert_gt_disps_to_depths_kitti(gt_disparities)

    for t_id in tqdm(range(num_test_images)):
        # Show the Disparity/Depth ground truths and the corresponding predictions for the evaluation images.
        try:
            if args.show_test_results:
                print("gt_disp:", gt_disparities[t_id])
                print(np.min(gt_disparities[t_id]), np.max(gt_disparities[t_id]))
                print()
                print("gt_depth:", gt_depths[t_id])
                print(np.min(gt_depths[t_id]), np.max(gt_depths[t_id]))
                print()
                print("pred:", pred_array[t_id])
                print(np.min(pred_array[t_id]), np.max(pred_array[t_id]))
                print()

                # TODO: Create Subplot
                plt.figure(100)
                plt.imshow(gt_disparities[t_id])
                plt.title('gt_disp')
                plt.draw()

                plt.figure(101)
                plt.imshow(gt_depths[t_id].astype(np.uint8))
                plt.title('gt_depth')
                plt.draw()

                plt.figure(102)
                plt.imshow(pred_array[t_id])
                plt.title('pred')
                plt.draw()

                plt.pause(0.001)

        except IndexError:
            break

    return gt_depths


def generate_depth_maps_eigen_split(pred_array, args_gt_path):
    num_test_images = 697

    test_files = read_text_lines(args.test_file_path)
    gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, args_gt_path)

    gt_depths = []
    print('\n[Metrics] Generating depth maps...')

    for t_id in tqdm(range(num_test_images)):
        try:
            camera_id = cams[t_id]  # 2 is left, 3 is right
            gt_depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, False)

            gt_depths.append(gt_depth.astype(np.float32))

            # Show the corresponding generated Depth Map from the Stereo Pair.
            if args.show_test_results:
                # print("depth:\n", gt_depth)
                # print("pred:\n", pred_array[t_id])

                # TODO: Create Subplot
                plt.figure(100)
                plt.imshow(imageio.imread(im_files[t_id]))
                plt.title('image')
                plt.draw()

                plt.figure(101)
                plt.imshow(gt_depth.astype(np.uint8))
                plt.title('gt_depth')
                plt.draw()

                plt.figure(102)
                plt.imshow(pred_array[t_id])
                plt.title('pred')
                plt.draw()

                plt.pause(0.001)

        except IndexError:
            break

    return gt_depths


def generate_depth_maps_eigen_continuous_split(pred_array, args_gt_path):
    num_test_images = 697

    test_files = read_text_lines(args.test_file_path)
    gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, args_gt_path)

    gt_depths = []
    gt_depths_continuous = []
    invalid_idx = []
    print('\n[Metrics] Generating depth maps...')

    for t_id in tqdm(range(num_test_images)):
        camera_id = cams[t_id]  # 2 is left, 3 is right
        gt_depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, False)
        gt_depths.append(gt_depth.astype(np.float32))

        depth_split = gt_files[t_id].split('/')
        new_depth_filename = depth_split[8] + '_' + depth_split[-1].replace('.bin', '.png')
        depth_replace = gt_files[t_id].replace('velodyne_points/data/', 'proc_kitti_nick/disp2/')
        depth_head, _ = os.path.split(depth_replace)
        gt_depth_continuous_path = os.path.join(depth_head, new_depth_filename)

        # print(gt_calib[t_id])
        # print(gt_files[t_id])
        # print(gt_depth_continuous_path)
        # print(im_sizes[t_id])

        try:
            gt_depth_continuous = imageio.imread(gt_depth_continuous_path).astype(
                np.float32) / 3.0  # Convert uint8 to float, meters
            gt_depths_continuous.append(gt_depth_continuous)
        except FileNotFoundError:
            gt_depths_continuous.append(np.zeros(shape=im_sizes[t_id]))
            invalid_idx.append(t_id)

        # Show the corresponding generated Depth Map from the Stereo Pair.
        if args.show_test_results:
            # print(imageio.imread(im_files[t_id]))
            # input("image")
            # print(depth)
            # print(np.min(depth), np.max(depth))
            # input("gt_depth")
            # print(gt_depth_continuous)
            # print(np.min(gt_depth_continuous), np.max(gt_depth_continuous))
            # print("gt_depth_continuous")
            # print(pred_array[t_id])
            # input("pred")

            # TODO: Create Subplot
            plt.figure(100)
            plt.imshow(imageio.imread(im_files[t_id]))
            plt.title('image')
            plt.draw()

            plt.figure(101)
            plt.imshow(gt_depth.astype(np.uint8))
            plt.title('gt_depth')
            plt.draw()

            plt.figure(102)
            plt.imshow(gt_depth_continuous)
            plt.title('gt_depth_continuous')
            plt.draw()

            plt.figure(103)
            plt.imshow(pred_array[t_id])
            plt.title('pred')
            plt.draw()

            plt.pause(0.001)

    print()
    print("%d missing files on continuous dataset." % len(invalid_idx))
    print("Missing indexes:", invalid_idx)
    print()

    return gt_depths


def generate_depth_maps(pred_list, gt_list, args_gt_path):
    """Generate Depth Maps for kitti_stereo, eigen, eigen_continuous splits."""
    pred_depths = np.array(pred_list)

    if args.test_split == 'kitti_stereo':
        gt_depths = generate_depth_maps_kitti_split(pred_depths, args_gt_path)

    elif args.test_split == 'eigen':
        gt_depths = generate_depth_maps_eigen_split(pred_depths, args_gt_path)

    elif args.test_split == 'eigen_continuous':
        gt_depths = generate_depth_maps_eigen_continuous_split(pred_depths, args_gt_path)

    else:
        gt_depths = np.array(gt_list)

    return pred_depths, gt_depths


# FIXME: Always saves with the index 0
def save_metrics_results_csv(metrics):
    """Logs the obtained simulation results on a .csv file."""
    save_metrics_filename = settings.output_dir + 'results_test_monodepth.csv'
    print("\n[Results] Logging simulation info to '%s' file..." % save_metrics_filename)

    if os.path.exists(save_metrics_filename):
        df_metrics = pd.DataFrame(metrics, index=[0])
        df_metrics.to_csv(save_metrics_filename, mode='a', header=False)
    else:
        df_metrics = pd.DataFrame(metrics, index=[0])
        df_metrics.to_csv(save_metrics_filename)

    print(df_metrics)


def stats_depth_txt2csv(num_evaluated_pairs):
    stats_depth_csv_filename = settings.output_dir + 'results_test_kitti_depth.csv'

    df = pd.read_csv(settings.output_tmp_pred_dir + 'stats_depth.txt', sep=':', header=None)
    df[0] = df[0].apply(lambda x: x.replace(' ', '_'))
    df[0] = df[0].apply(lambda x: x.replace('__', '_'))
    df = df.T
    df = df.rename(columns=df.iloc[0])
    df = df.drop(df.index[0])
    df.drop(df.columns[0], axis=1)
    df['model'] = args.model_path
    df = df.set_index('model')

    # Adding New info
    df['test_split'] = args.test_file_path
    df['num_test_images'] = num_evaluated_pairs

    # Rearranges Columns Order
    new_order = [27, 28] + list(range(27))
    df = df[df.columns[new_order]]

    if not os.path.isfile(stats_depth_csv_filename):
        df.to_csv(stats_depth_csv_filename)
    else:
        df.to_csv(stats_depth_csv_filename, mode='a', header=False)


@deprecated(version='1.0',
            reason="You shouldn't use this function. It's not recommended to evaluate the trained methods on ground truth images generated from LIDAR measurements.")
def evaluation_tool_monodepth(pred_depths, gt_depths):
    num_test_images = len(pred_depths)

    rms = np.zeros(num_test_images, np.float32)
    log_rms = np.zeros(num_test_images, np.float32)
    abs_rel = np.zeros(num_test_images, np.float32)
    sq_rel = np.zeros(num_test_images, np.float32)
    d1_all = np.zeros(num_test_images, np.float32)
    a1 = np.zeros(num_test_images, np.float32)
    a2 = np.zeros(num_test_images, np.float32)
    a3 = np.zeros(num_test_images, np.float32)

    print('[Metrics] Computing metrics...')
    for i in tqdm(range(num_test_images)):
        try:
            if args.test_split == 'eigen_continuous':  # FIXME: Remove everything related to eigen_continuous
                gt_depth = gt_depths_continuous[i]

                if i in invalid_idx:
                    print("invalid")
                    continue  # Skips the rest of the code
            else:
                gt_depth = gt_depths[i]

            pred_depth = pred_depths[i]

            pred_depth[pred_depth < args.min_depth] = args.min_depth
            pred_depth[pred_depth > args.max_depth] = args.max_depth

            if args.test_split == 'eigen' or args.test_split == 'eigen_continuous':
                mask = np.logical_and(gt_depth > args.min_depth, gt_depth < args.max_depth)

                if args.garg_crop or args.eigen_crop:
                    gt_height, gt_width = gt_depth.shape

                    # Crop used by Garg ECCV16
                    # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
                    if args.garg_crop:
                        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                         0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
                    # Crop we found by trial and error to reproduce Eigen NIPS14 results
                    elif args.eigen_crop:
                        crop = np.array([0.3324324 * gt_height, 0.91351351 * gt_height,
                                         0.0359477 * gt_width, 0.96405229 * gt_width]).astype(np.int32)

                    crop_mask = np.zeros(mask.shape)
                    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                    mask = np.logical_and(mask, crop_mask)

            mask = gt_depth > 0.0

            abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask],
                                                                                            pred_depth[mask])

            # Show gt/pred Images
            if False:  # TODO: Remover?
                # TODO: Create Subplot
                plt.figure(102)
                plt.title('pred')
                plt.imshow(pred_depth)

                plt.figure(103)
                plt.imshow(gt_depth)
                plt.title('left')

                plt.draw()
                plt.pause(1)

        except IndexError:
            break

    # --------- #
    #  Results  #
    # --------- #
    test_split = args.dataset if args.test_split == '' else args.test_split
    metrics = {
        'model': args.model_path,
        'test_split': test_split,
        'num_test_images': num_test_images,
        'cap': args.max_depth,
        'abs_rel': abs_rel.mean(),
        'sq_rel': sq_rel.mean(),
        'rms': rms.mean(),
        'log_rms': log_rms.mean(),
        'd1_all': d1_all.mean(),
        'a1': a1.mean(),
        'a2': a2.mean(),
        'a3': a3.mean()}

    # Save results on .csv file
    save_metrics_results_csv(metrics)

    # Display Results
    print()
    print("# ----------------- #")
    print("#  Metrics Results  #")
    print("# ----------------- #")
    print(metrics)


def evaluation_tool_kitti_depth(num_test_images):
    import subprocess

    print("[KITTI Depth Evaluation Tool] Invoking 'evaluate_depth' executable...")
    subprocess.call(
        [r"modules/evaluation/kitti_depth_prediction_devkit/cpp/evaluate_depth",
         "{}".format(settings.output_tmp_gt_dir),
         "{}".format(settings.output_tmp_pred_dir)])

    stats_depth_txt2csv(num_test_images)
