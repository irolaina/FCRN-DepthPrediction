import imageio
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from common import settings
from modules.args import args
from .evaluation_utils import *


# ===========
#  Functions
# ===========
def generate_depth_maps(pred_list, gt_list, args_gt_path):
    pred_array = np.array(pred_list)
    gt_array = np.array(gt_list)

    if args.test_split == 'kitti':
        num_samples = 200

        # The FCRN predicts meters instead of disparities, so it's not necessary to convert disps to depth!!!
        gt_disparities = load_gt_disp_kitti(args_gt_path)

        print('\n[Metrics] Generating depth maps...')
        gt_depths = convert_gt_disps_to_depths_kitti(gt_disparities)

        for t_id in tqdm(range(num_samples)):
            # Show the Disparity/Depth ground truths and the corresponding predictions for the evaluation images.
            if False:  # TODO: ativar pela flag -u
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
                plt.draw()

                plt.figure(101)
                plt.imshow(gt_depths[t_id].astype(np.uint8))
                plt.title('gt_depth')
                plt.draw()

                plt.figure(102)
                plt.imshow(pred_array[t_id])
                plt.title('pred')
                plt.draw()

                plt.pause(1)
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

            # Show the corresponding generated Depth Map from the Stereo Pair.
            if False:  # TODO: ativar pela flag -u
                print(depth)
                input("depth")
                print(pred_array[t_id])
                input("pred")

                plt.figure(100)
                plt.imshow(imageio.imread(im_files[t_id]))
                plt.title('image')
                plt.draw()

                plt.figure(101)
                plt.imshow(depth.astype(np.uint8))
                plt.title('gt_depth')
                plt.draw()

                plt.figure(102)
                plt.imshow(pred_array[t_id])
                plt.title('pred')
                plt.draw()

                plt.pause(1)
                input("Continue...")

    elif args.test_split == 'eigen_continuous':
        num_samples = 697
        test_files = read_text_lines(args.test_file_path)
        gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, args_gt_path)

        num_test = len(im_files)
        gt_depths = []
        gt_depths_continuous = []
        pred_depths = []
        invalid_idx = []
        print('\n[Metrics] Generating depth maps...')
        # num_samples = 5  # Only for testing!
        for t_id in tqdm(range(num_samples)):
            camera_id = cams[t_id]  # 2 is left, 3 is right
            depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True)
            gt_depths.append(depth.astype(np.float32))

            depth_split = gt_files[t_id].split('/')
            new_depth_filename = depth_split[8] + '_' + depth_split[-1].replace('.bin', '.png')
            depth_replace = gt_files[t_id].replace('velodyne_points/data/', 'proc_kitti_nick/disp2/')
            depth_head, depth_tail = os.path.split(depth_replace)
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
            if False:  # TODO: ativar pela flag -u
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

                plt.figure(100)
                plt.imshow(imageio.imread(im_files[t_id]))
                plt.title('image')
                plt.draw()

                plt.figure(101)
                plt.imshow(depth.astype(np.uint8))
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

                plt.pause(1)
                input("Continue...")

        print()
        print("%d missing files on continuous dataset." % len(invalid_idx))
        print("Missing indexes:", invalid_idx)
        print()
    else:
        num_samples = len(pred_array)
        gt_depths = gt_array

    pred_depths = pred_array

    return pred_depths, gt_depths, num_samples

def save_stats_depth_to_csv():

    stats_depth_csv_filename = settings.output_dir + 'kitti_depth_eval.csv'

    df = pd.read_csv(settings.output_tmp_pred_dir + 'stats_depth.txt', sep=':', header=None)
    df[0] = df[0].apply(lambda x: x.replace(' ', '_'))
    df[0] = df[0].apply(lambda x: x.replace('__', '_'))
    df = df.T
    df = df.rename(columns=df.iloc[0])
    df = df.drop(df.index[0])
    df.drop(df.columns[0], axis=1)
    df['model'] = args.model_path
    df = df.set_index('model')

    if not os.path.isfile(stats_depth_csv_filename):
        df.to_csv(stats_depth_csv_filename)
    else:
        with open(stats_depth_csv_filename, 'a') as file:
            df.to_csv(file, header=False)

def evaluate(pred_list, gt_list, args_gt_path, evaluation_tool='monodepth'):
    # --------------------------------------------- #
    #  Generate Depth Maps for kitti, eigen splits  #
    # --------------------------------------------- #
    pred_depths, gt_depths, num_samples = generate_depth_maps(pred_list, gt_list,
                                                              args_gt_path)  # FIXME: Talvez esta função não precise estar dentro da evaluate()

    # ----------------- #
    #  Compute Metrics  #
    # ----------------- #
    if evaluation_tool == 'monodepth':
        rms = np.zeros(num_samples, np.float32)
        log_rms = np.zeros(num_samples, np.float32)
        abs_rel = np.zeros(num_samples, np.float32)
        sq_rel = np.zeros(num_samples, np.float32)
        d1_all = np.zeros(num_samples, np.float32)
        a1 = np.zeros(num_samples, np.float32)
        a2 = np.zeros(num_samples, np.float32)
        a3 = np.zeros(num_samples, np.float32)

        print('[Metrics] Computing metrics...')
        for i in tqdm(range(num_samples)):

            if args.test_split == 'eigen_continuous':
                gt_depth = gt_depths_continuous[i]

                if i in invalid_idx:
                    print("invalid")
                    continue  # Skips the rest of the code
            else:
                gt_depth = gt_depths[i]

            pred_depth = pred_depths[i]

            pred_depth[pred_depth < args.min_depth] = args.min_depth
            pred_depth[pred_depth > args.max_depth] = args.max_depth

            if args.test_split == 'eigen' or args.test_split == 'eigen_continuous':  # TODO: Validar!
                mask = np.logical_and(gt_depth > args.min_depth, gt_depth < args.max_depth)

                if args.garg_crop or args.eigen_crop:
                    gt_height, gt_width = gt_depth.shape

                    # crop used by Garg ECCV16
                    # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
                    if args.garg_crop:
                        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                         0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
                    # crop we found by trial and error to reproduce Eigen NIPS14 results
                    elif args.eigen_crop:
                        crop = np.array([0.3324324 * gt_height, 0.91351351 * gt_height,
                                         0.0359477 * gt_width, 0.96405229 * gt_width]).astype(np.int32)

                    crop_mask = np.zeros(mask.shape)
                    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                    mask = np.logical_and(mask, crop_mask)

            mask = gt_depth > 0

            abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask],
                                                                                            pred_depth[mask])

            # Show gt/pred Images
            if False:  # TODO: Remover?
                plt.figure(102)
                plt.title('pred')
                plt.imshow(pred_depth)

                plt.figure(103)
                plt.imshow(gt_depth)
                plt.title('left')

                plt.draw()
                plt.pause(1)

        # End of Loop
        metrics = {'abs_rel': abs_rel.mean(),
                   'sq_rel': sq_rel.mean(),
                   'rms': rms.mean(),
                   'log_rms': log_rms.mean(),
                   'd1_all': d1_all.mean(),
                   'a1': a1.mean(),
                   'a2': a2.mean(),
                   'a3': a3.mean()}

        # --------- #
        #  Results  #
        # --------- #
        # Save results on .csv file
        test_split = args.dataset if args.test_split == '' else args.test_split

        results_metrics = [
            args.model_path,
            test_split,
            metrics['abs_rel'],
            metrics['sq_rel'],
            metrics['rms'],
            metrics['log_rms'],
            metrics['d1_all'],
            metrics['a1'],
            metrics['a2'],
            metrics['a3']]

        # TODO: Mover para utils.py?
        def save_metrics_results_csv():
            """Logs the obtained simulation results on a .csv file."""
            save_file_path = settings.output_dir + 'results_metrics.csv'
            print("\n[Results] Logging simulation info to '%s' file..." % save_file_path)

            df_results_metrics = pd.read_csv(save_file_path)

            results_series = pd.Series(results_metrics)

            df_results_metrics.append(results_series, ignore_index=True)
            df_results_metrics.to_csv(save_file_path)

            print(df_results_metrics)

        save_metrics_results_csv()

        # Display Results
        results_header_formatter = "{:>20}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}"
        results_data_formatter = "{:>20}, {:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}"

        print()
        print("# ----------------- #")
        print("#  Metrics Results  #")
        print("# ----------------- #")
        print(args.model_path)
        print(
            results_header_formatter.format('split', 'abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'd1', 'd2', 'd3'))

        print(results_data_formatter.format(*results_metrics[1:]))

    elif evaluation_tool == 'kitti_depth':
        import subprocess

        print("[KittiDepth Evaluation Tool] Invoking 'evaluate_depth' executable...")
        subprocess.call(
            [r"evaluation/kitti_depth_prediction_devkit/cpp/evaluate_depth", "{}".format(settings.output_tmp_gt_dir),
             "{}".format(settings.output_tmp_pred_dir)])

        save_stats_depth_to_csv()

    else:
        raise SystemError
