#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ====== #
#  FCRN  #
# ====== #
# Attention! The maximum range that the Network can predict depends on the maximum distance recorded in the dataset.

# =======
#  To-Do
# =======
# Must do
# [Dataset] TODO: Verificar se aquela imagem do Apolloscape estava realmente corrompida
# [Dataset] TODO: Caso ela realmente estiver corrompida no .zip, enviar e-mail para Apolloscape
# [Dataset] FIXME: Aparentemente existe uma série de imagens inválidas no dataset apolloscape. Use scripts/check_apolloscape_imgs.py

# [Valid] FIXME: valid.loss sempre igual a zero quando utiliza-se a as flags 'valid' e 'eigen_grads'

# [Test] FIXME: A Terceira imagem de Test, a depth_resized (~20m) não possui o mesmo range que a depth image (~70 m). Reproduce: python3 predict_nick.py -m test -s kitti_continuous --px all -r output/fcrn/kitti_continuous/all_px/eigen_grads/2018-06-27_11-14-21/restore/model.fcrn -u

# Known Bugs
# [Train] FIXME: Early Stopping parece não disparar.
# [Train] FIXME: Resolver erro que acontece com as imagens do ApolloScape durante valid evaluation @ ~24000
# [All] TODO: Devo continuar usando tf.image.resize_images()? Há relatos desta função ser bugada

# Optional
# [Dataset] FIXME: Descobrir porquê o código do vitor (cnn_hilbert) não está gerando todas as imagens (disp1 e disp2)
# [Train] TODO: Dar suporte ao Make3D
# [Train] TODO: Adicionar feature para permitir fine tunning

# Ideas
# TODO: Trabalhar com Sequências Temporais: Semelhante à SfM, LSTM
# TODO: Como Monocular Depth pode auxiliar em Visual Odometry?
# TODO: O trabalho "Sparsity Invariant CNNs" diz que redes neurais devem ser capazes de distinguir pixeis observados e pixeis inválidos. Não simplesmente "mask them out".
# TODO: O trabalho "Deep Ordinal Regression Network (DORN) for Monocular Depth Estimation" aponta o problema com as arquitetura clássicas de MDE que inicialmente foram desenvolvidas para Image Recognition, cujas operações de max-pooling e striding reduzem a resolução espacial dos features maps para este tipo de aplicação
# TODO: Investigar Redes Neurais que estudam esparsidade DENTRO das redes e nas ENTRADAS. Ref: "Sparsity Invariant CNNs"
# TODO: De acordo com DORN, abordar o problema de estimação como um problema multi-class classification é mais indicado do que tratá-lo como um problema de regressão
# TODO: A Rede da Laina possui Weight Decay?

# ===========
#  Libraries
# ===========
import os
import sys
import time
import warnings

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import exposure, img_as_uint
from tqdm import tqdm

# Custom Libraries
from modules import metrics
from modules.args import args
from modules.dataloader import Dataloader
from modules.framework import load_model, SessionWithExitSave, Model
from modules.plot import Plot
from modules.test import Test
from modules.third_party.laina.fcrn import ResNet50UpProj
from modules.utils import settings

# ==========================
#  [Train] Framework Config
# ==========================
TRAIN_ON_SINGLE_IMAGE = False  # Default: False
ENABLE_EARLY_STOP = True  # Default: True
ENABLE_TENSORBOARD = True  # Default: True

# =========================
#  [Test] Framework Config
# =========================
# Select Subset:
evaluation_subset_selector = {0: 'test', 1: 'train'}
SELECTED_EVALUATION_SUBSET = evaluation_subset_selector[0]

SAVE_TEST_DISPARITIES = True  # Default: True

# ==================
#  Global Variables
# ==================
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # Limits Tensorflow to see only the specified GPU.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")  # Suppress Warnings

# ========= #
#  Testing  #
# ========= #
def test():
    # Local Variables
    num_test_images = None

    # -----------------------------------------
    #  Network Testing Model - Importing Graph
    # -----------------------------------------
    # Loads the dataset and restores a specified trained model.
    data = Dataloader()

    # Searches dataset images filenames
    if SELECTED_EVALUATION_SUBSET == 'test':
        _, _, _, _, num_test_images = data.get_test_data(test_split=args.test_split, test_file_path=args.test_file_path)
    elif SELECTED_EVALUATION_SUBSET == 'train':
        data.test_image_filenames, data.test_depth_filenames, _, _, num_test_images = data.get_train_data()

    model = Test(data)

    with tf.Session() as sess:
        print('\n[network/Testing] Loading the model...')
        load_model(saver=tf.train.Saver(), sess=sess)

        # ==============
        #  Testing Loop
        # ==============
        pred_list, gt_list = [], []

        if args.debug:
            num_test_images = 5  # Only for testing!

        # TODO: Criar uma classe de test assim como fiz para train e valid, e declarar este objeto dentro dela
        if args.show_test_results:
            test_plot_obj = Plot(args.mode, title='Test Predictions')

        print("\n[Network/Testing] Generating Predictions...")
        timer = -time.time()
        for i in tqdm(range(num_test_images)):
            timer2 = -time.time()

            # Evalute the network for the given image
            # data.test_depth_filenames = [] # Only for testing the following condition!!! # FIXME: Atualmente, o código não dá suporte para esta situação
            if data.test_depth_filenames:  # It's not empty
                feed_test = {model.tf_image_key: data.test_image_filenames[i],
                             model.tf_depth_key: data.test_depth_filenames[i]}

                _, depth, depth_resized = sess.run(model.depth_op, feed_test)

            else:
                feed_test = {model.tf_image_key: data.test_image_filenames[i]}

            _, image, image_resized, pred, pred_up = sess.run(model.image_op + model.pred_op, feed_test)

            # Clips Predictions at 50, 80 meters
            try:
                pred_50, pred_80 = sess.run([model.tf_pred_50, model.tf_pred_80], feed_test)
            except AttributeError:
                pred_50 = np.zeros((model.batch_size,) + model.output_size.get_size())
                pred_80 = np.zeros((model.batch_size,) + model.output_size.get_size())

            # Fill arrays for later on metrics evaluation
            if args.eval_tool == 'monodepth':
                pred_list.append(pred_up[0, :, :, 0])
                gt_list.append(depth[:, :, 0])

            # Saves the Test Predictions as uint16 PNG Images
            if SAVE_TEST_DISPARITIES:
                # Convert the Predictions Images from float32 to uint16
                pred_up_uint16 = img_as_uint(exposure.rescale_intensity(pred_up[0], out_range='float'))
                depth_uint16 = img_as_uint(exposure.rescale_intensity(depth, out_range='float'))

                # Save PNG Images
                imageio.imsave(settings.output_tmp_pred_dir + 'pred' + str(i) + '.png', pred_up_uint16)
                imageio.imsave(settings.output_tmp_gt_dir + 'gt' + str(i) + '.png', depth_uint16)

            # Prints Testing Progress
            timer2 += time.time()
            # print('step: %d/%d | t: %f | size(pred_list+gt_list): %d' % (i + 1, num_test_images, timer2, sys.getsizeof(pred_list)+sys.getsizeof(gt_list)))

            # Show Results
            if args.show_test_results:
                # TODO: Fazer um lista the 'images_to_display' e dar append das imagens na lista
                test_plot_obj.show_test_results(image=image,
                                                depth=depth[:, :, 0],
                                                image_resized=image_resized,
                                                depth_resized=depth_resized[:, :, 0],
                                                pred=pred[0, :, :, 0],
                                                pred_up=pred_up[0, :, :, 0],
                                                pred_50=pred_50[0, :, :, 0],
                                                pred_80=pred_80[0, :, :, 0],
                                                i=i + 1)

            # input("Continue...")

        # Testing Finished.
        timer += time.time()
        print("[Network/Testing] Testing FINISHED! Time elapsed: {} s\n".format(timer))

        # =========
        #  Results
        # =========
        # Calculates Metrics
        if data.test_depth_filenames:
            print("[Network/Testing] Calculating Metrics based on Test Predictions...")

            print('args.test_split:', args.test_split)
            print('args.test_file_path:', args.test_file_path)
            print('dataset_path:', data.dataset.dataset_path)
            print()

            # Invokes Evaluation Tools
            if args.eval_tool == 'monodepth':
                pred_depths, gt_depths = metrics.generate_depth_maps(pred_list, gt_list, data.dataset.dataset_path)
                metrics.evaluation_tool_monodepth(pred_depths, gt_depths)
            elif args.eval_tool == 'kitti_depth':
                metrics.evaluation_tool_kitti_depth(num_test_images)
            else:
                raise SystemError(
                    "Invalid 'eval_tool' selected. Choose one of the options: 'monodepth' or 'kitti_depth'.")

        else:
            print(
                "[Network/Testing] It's not possible to calculate metrics. There are no corresponding labels for generated predictions!")


# ========= #
#  Predict  #
# ========= #
def predict():
    # Default input size
    batch_size, height, width = 1, 228, 304

    # Read image (uint8)
    img = Image.open(args.image_path)
    img = np.array(img)

    # ------- #
    #  Graph  #
    # ------- #
    # Create a placeholder for the input image
    tf_image = tf.placeholder(tf.uint8, shape=(None, None, 3))
    tf_image_float32 = tf.image.convert_image_dtype(tf_image, tf.float32)  # uint8 -> float32 [0.0, 1.0]
    tf_image_resized = tf.image.resize_images(tf_image_float32, [height, width], method=tf.image.ResizeMethod.AREA,
                                              align_corners=True)

    tf_image_resized_uint8 = tf.image.convert_image_dtype(tf_image_resized, tf.uint8)  # Visual purpose

    with tf.variable_scope('model'):
        # Construct the network
        net = ResNet50UpProj({'data': tf.expand_dims(tf_image_resized, axis=0)}, batch=batch_size, keep_prob=1,
                             is_training=False)
        tf_pred = net.get_output()
        tf_pred_up = tf.image.resize_images(tf_pred, img.shape[:2], tf.image.ResizeMethod.BILINEAR,
                                            align_corners=True)
        tf_imask_80 = tf.where(tf_pred_up < 80.0, tf.ones_like(tf_pred_up), tf.zeros_like(tf_pred_up))
        tf_pred_up_80 = tf.multiply(tf_pred_up, tf_imask_80)

        # for var in tf.trainable_variables():
        #     print(var)

    # Merge Ops
    pred_op = [tf_image, tf_image_resized_uint8, tf_pred, tf_pred_up, tf_pred_up_80]

    # Print Variables
    # print(img)
    print(img.shape, img.dtype)

    print(tf_image)
    print(tf_image_resized)
    print(tf_pred)

    with tf.Session() as sess:
        # --------- #
        #  Restore  #
        # --------- #
        print('\n[network/Predicting] Loading the model...')
        load_model(saver=tf.train.Saver(), sess=sess)

        # ----- #
        #  Run  #
        # ----- #
        # Evalute the network for the given image
        image, image_resized_uint8, pred, pred_up, pred_up_80 = sess.run(pred_op, feed_dict={tf_image: img})

        # --------- #
        #  Results  #
        # --------- #
        fig = plt.figure(figsize=(15, 3))
        fig.subplots_adjust(bottom=0.025, left=0.025, top=0.975, right=0.975)
        X = [(1, 4, (1, 2)), (1, 4, 3), (1, 4, 4)]
        axes = []
        for nrows, ncols, plot_number in X:
            axes.append(fig.add_subplot(nrows, ncols, plot_number))

        axes[0].imshow(image)
        axes[1].imshow(image_resized_uint8)
        img2 = axes[2].imshow(pred[0, :, :, 0])

        axes[0].set_title('Image')
        axes[1].set_title('Resized')
        axes[2].set_title('Pred')

        # Fix Colorbar size
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="5%", pad=0.15)
        fig.colorbar(img2, cax=cax)

        plt.figure(2)
        plt.imshow(pred_up[0, :, :, 0])
        plt.imsave(settings.output_dir + 'pred_up.png', pred_up[0, :, :, 0])

        plt.figure(3)
        plt.imshow(pred_up_80[0, -264:, :, 0])
        plt.imsave(settings.output_dir + 'pred_up_80.png', pred_up_80[0, -264:, :, 0])
        plt.show()

        return pred

def read_text_file(filename):
    dataset_path = '/media/nicolas/nicolas_seagate/datasets/kitti/'

    print("\n[Dataloader] Loading '%s'..." % filename)
    try:
        data = np.genfromtxt(filename, dtype='str', delimiter='\t')
        # print(data.shape)

        # Parsing Data
        depth_continuous_filenames = list(data[:, 0])
        depth_semidense_filenames = list(data[:, 1])

        timer = -time.time()
        depth_continuous_filenames = [dataset_path + filename for filename in depth_continuous_filenames]
        depth_semidense_filenames = [dataset_path + filename for filename in depth_semidense_filenames]
        timer += time.time()
        print('time:', timer, 's\n')

    except OSError:
        raise OSError("Could not find the '%s' file." % filename)

    return depth_continuous_filenames, depth_semidense_filenames

def read_hilbert_maps_depth_image(filename):
    return imageio.imread(filename).astype('float32') / 3.0

def read_kitti_depth_depth_image(filename):
    return imageio.imread(filename).astype('float32') / 256.0


def imsave_as_uint16_png(filename, image_float32):
    # Converts the Predictions Images from float32 to uint16 and Saves as PNG Images

    image_uint16 = img_as_uint(exposure.rescale_intensity(image_float32, out_range='float'))
    imageio.imsave(filename, image_uint16)

def evaluate_hilbert_maps_on_kitti_depth():
    showImages = False
    # eval_tool = 'monodepth'
    eval_tool = 'kitti_depth'

    # Loads split file containing Hilbert Maps and KITTI Depth filenames
    hilbert_maps_filenames, kitti_depth_filenames = read_text_file('data/new_splits/eigen_split_based_on_kitti_depth/eigen_test_kitti_depth_aligned_with_kitti_continuous_files.txt')
    assert len(kitti_depth_filenames) == len(hilbert_maps_filenames)

    print(len(kitti_depth_filenames), len(hilbert_maps_filenames))

    # Read Images
    hilbert_maps_depths = []
    kitti_depth_dephts = []
    num_test_images = len(kitti_depth_filenames)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    for i, (hilbert_maps_filename, kitti_depth_filename) in enumerate(list(zip(hilbert_maps_filenames, kitti_depth_filenames))):
        hilbert_maps_depth = read_hilbert_maps_depth_image(hilbert_maps_filename)  # Continuous
        kitti_depth_depth = read_kitti_depth_depth_image(kitti_depth_filename)  # Semi-Dense

        hilbert_maps_depths.append(hilbert_maps_depth)
        kitti_depth_dephts.append(kitti_depth_depth)

        if showImages:
            ax1.imshow(hilbert_maps_depth)
            ax2.imshow(kitti_depth_depth)
            plt.draw()
            plt.pause(0.001)

        # Saves the Test Predictions as uint16 PNG Images
        if SAVE_TEST_DISPARITIES or eval_tool == 'monodepth':
            imsave_as_uint16_png(settings.output_tmp_pred_dir + 'pred' + str(i) + '.png', hilbert_maps_depth)
            imsave_as_uint16_png(settings.output_tmp_gt_dir + 'gt' + str(i) + '.png', kitti_depth_depth)

        print('{}/{}'.format(i, num_test_images))

    # Invokes Evaluation Tools
    if eval_tool == 'monodepth':
        metrics.evaluation_tool_monodepth(hilbert_maps_depths, kitti_depth_dephts)
    elif eval_tool == 'kitti_depth':
        metrics.evaluation_tool_kitti_depth(num_test_images)
    else:
        raise SystemError("Invalid 'eval_tool' selected. Choose one of the options: 'monodepth' or 'kitti_depth'.")


# ======
#  Main
# ======
if __name__ == '__main__':
    evaluate_hilbert_maps_on_kitti_depth()

    print("\nDone.")
    sys.exit()


