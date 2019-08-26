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

    parser.add_argument('--debug', action='store_true', help="Enables the Debug Mode", default=False)

    # Input
    parser.add_argument('--machine', type=str, help="Selects the current Machine: 'olorin', 'nicolas', etc",
                        default='nicolas')

    parser.add_argument('-m', '--mode', type=str, help="Selects 'train' or 'test' mode", default='train')

    parser.add_argument('--model_name', type=str, help="Selects Network topology: 'fcrn', etc",
                        default='fcrn')

    parser.add_argument('--gpu', type=str, help="Selects which gpu to run the code", default='0')

    # ========== #
    #  Training  #
    # ========== #

    parser.add_argument('--retrain', action='store_true', help="Enables the Retrain Mode", default=False)

    parser.add_argument('-s', '--dataset', action='store',
                        help="Selects the dataset: 'apolloscape', 'kitti_depth', 'kitti_discrete', 'kitti_continuous', 'nyudepth', or 'lrmjose'",
                        default='')

    parser.add_argument('--px', action='store',
                        help="Selects which pixels to optimize: 'all' or 'valid'", default='valid')

    parser.add_argument('--loss', type=str,
                        help="Selects the desired loss function: 'mse', 'berhu', 'eigen', 'eigen_grads', etc",
                        default='berhu')
    parser.add_argument('--batch_size', type=int, help="Defines the training batch size", default=4)
    parser.add_argument('--max_steps', type=int, help="Defines the max number of training steps", default=300000)
    parser.add_argument('-l', '--learning_rate', type=float, help="Defines the initial value of the learning rate",
                        default=1e-4)

    parser.add_argument('-d', '--dropout', type=float, help="Enables dropout in the model during training", default=1.0)
    parser.add_argument('--ldecay', action='store_true', help="Enables learning decay", default=False)
    parser.add_argument('-n', '--l2norm', action='store_true', help="Enables L2 Normalization", default=False)
    parser.add_argument('--data_aug', action='store_true', help="Enables Data Augmentation", default=True)

    parser.add_argument('--remove_sky', action='store_true', help="Removes sky for KITTI Datasets", default=False)

    parser.add_argument('--full_summary', action='store_true',
                        help="If set, it will keep more data for each summary. Warning: the file can become very large")

    parser.add_argument('--log_directory', type=str, help="Sets the directory to save checkpoints and summaries",
                        default='log_tb/')

    parser.add_argument('-t', '--show_train_progress', action='store_true', help="Shows Training Images progress.",
                        default=False)

    parser.add_argument('-v', '--show_valid_progress', action='store_true', help="Shows Validation Images progress.",
                        default=False)

    # ========= #
    #  Testing  #
    # ========= #
    parser.add_argument('--eval_tool', type=str,
                        help="Selects the evaluation tool for computing metrics: 'monodepth' or 'kitti_depth'",
                        default='')

    parser.add_argument('--test_split', type=str,
                        help="Selects the desired test split for State-of-art evaluation: 'kitti_stereo', 'eigen', 'eigen_continuous', etc",
                        default='')

    parser.add_argument('--test_file_path', type=str,
                        help="Evaluates the model for the speficied images from a test_file.txt file",
                        default='')

    parser.add_argument('--min_depth', type=float, help='Specifies the minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth', type=float, help='Specifies the maximum depth for evaluation', default=80)
    parser.add_argument('--eigen_crop', action='store_true', help='If set, crops according to Eigen NIPS14')
    parser.add_argument('--garg_crop', action='store_true', help='If set, crops according to Garg  ECCV16')

    parser.add_argument('-o', '--output_directory', type=str,
                        help='Sets the output directory for test disparities, if empty outputs to checkpoint folder',
                        default='')

    parser.add_argument('-u', '--show_test_results', action='store_true',
                        help="Shows the network predictions for the specified test split images", default=False)

    # ============ #
    #  Prediction  #
    # ============ #
    parser.add_argument('-r', '--model_path', type=str, help="Sets the path to a specific model to be restored",
                        default='')
    parser.add_argument('-i', '--image_path', help='Sets the path to the image to be predicted', default='')

    return parser.parse_args()


args = argument_handler()
