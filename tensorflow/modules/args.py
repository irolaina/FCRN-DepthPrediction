# ===========
#  Libraries
# ===========
import argparse


# ===========
#  Functions
# ===========
def argument_handler():
    # Creating Arguments Parser
    parser = argparse.ArgumentParser(
        "Train the FCRN (Fully Convolution Residual Network) Tensorflow implementation taking image files as input.")

    # Input
    parser.add_argument('--gpu', type=str, help="Selects which gpu to run the code", default='0')
    parser.add_argument('-m', '--mode', type=str, help="Selects 'train' or 'test' mode", default='train')

    # ========== #
    #  Training  #
    # ========== #
    parser.add_argument('--machine', type=str, help="Selects the current Machine: 'olorin', 'nicolas', etc",
                        default='nicolas')

    parser.add_argument('--model_name', type=str, help="Selects Network topology: 'fcrn', etc",
                        default='fcrn')

    parser.add_argument('-s', '--dataset', action='store',
                        help="Selects the dataset ['apolloscape', 'kittidepth', 'kitti_discrete', 'kitti_continuous', 'kitti_continuous_residential', 'nyudepth']",
                        default='')

    parser.add_argument('--px', action='store',
                        help="Selects which pixels to minimize ['all' or 'valid']", default='all')

    parser.add_argument('--data_aug', action='store_true', help="Enables Data Augmentation", default=True)

    parser.add_argument('--loss', type=str, help="Selects the desired loss function: 'mse', 'berhu', 'eigen', 'eigen_grads' etc",
                        default='berhu')

    parser.add_argument('--batch_size', type=int, help="Defines the Training batch size", default=4)
    parser.add_argument('--max_steps', type=int, help="Defines the number of max Steps", default=1000)
    parser.add_argument('-l', '--learning_rate', type=float, help="Defines the initial learning rate", default=1e-4)
    parser.add_argument('-d', '--dropout', type=float, help="Enables dropout in the model during training", default=0.5)
    parser.add_argument('--ldecay', action='store_true', help="Enables learning decay", default=False)
    parser.add_argument('-n', '--l2norm', action='store_true', help="Enables L2 Normalization", default=False)
    parser.add_argument('--remove_sky', action='store_true', help="Removes Sky for Kitti Datasets", default=False)

    parser.add_argument('--full_summary', action='store_true',
                        help="If set, it will keep more data for each summary. Warning: the file can become very large")

    parser.add_argument('--log_directory', type=str, help="Sets the directory to save checkpoints and summaries",
                        default='log_tb/')

    parser.add_argument('-t', '--show_train_progress', action='store_true', help="Shows Training Progress Images",
                        default=False)

    parser.add_argument('-v', '--show_valid_progress', action='store_true', help="Shows Validation Progress Images",
                        default=False)

    parser.add_argument('--test_split', type=str, help="Selects the desired test split for State-of-art evaluation: 'kitti_stereo', 'eigen', 'eigen_continuous', etc",
                        default='')

    parser.add_argument('--eval_tool', type=str, help="Selects the evaluation tool for computing metrics: 'monodepth' or 'kitti_depth'",
                        default='kitti_depth')

    parser.add_argument('--test_file_path', type=str, help="Evaluates the Model for the images speficied by test_file.txt file",
                        default='')

    parser.add_argument('--min_depth', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth', type=float, help='maximum depth for evaluation', default=80)
    parser.add_argument('--eigen_crop', help='If set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--garg_crop', help='If set, crops according to Garg  ECCV16', action='store_true')

    # ========= #
    #  Testing  #
    # ========= #
    parser.add_argument('-o', '--output_directory', type=str,
                        help='Sets the output directory for test disparities, if empty outputs to checkpoint folder',
                        default='')

    parser.add_argument('-u', '--show_test_results', action='store_true',
                        help="Show the first batch testing Network prediction img", default=False)

    # ============ #
    #  Prediction  #
    # ============ #
    parser.add_argument('-r', '--model_path', type=str, help="Sets the path to a specific model to be restored", default='')
    parser.add_argument('-i', '--image_path', help='Sets the path to the image to be predicted', default='')

    parser.add_argument('--debug', action='store_true', help="Enables the Debug Mode", default=False)

    return parser.parse_args()


args = argument_handler()
