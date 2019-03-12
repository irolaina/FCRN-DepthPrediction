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


# ===================== #
#  Training/Validation  #
# ===================== #
def train():
    # ----------------------------------------- #
    #  Network Training Model - Building Graph  #
    # ----------------------------------------- #
    graph = tf.Graph()
    with graph.as_default():
        data = Dataloader()

        # If enabled, the framework will train the network on only one image!!!
        if TRAIN_ON_SINGLE_IMAGE:
            data.train_image_filenames = np.expand_dims(data.train_image_filenames[0], axis=0)
            data.train_depth_filenames = np.expand_dims(data.train_depth_filenames[0], axis=0)

        # Build Network Model
        model = Model(data)
        model.collect_summaries(graph)
        model.create_train_saver()

    # ---------------------------------------- #
    #  Network Training Model - Running Graph  #
    # ---------------------------------------- #
    # Local Variables and Memory Allocation
    epoch, step = 0, 0

    # Proclaim the epochs
    max_epochs = int(np.floor(args.batch_size * args.max_steps / data.num_train_samples))
    print('\nTrain with approximately %d epochs' % max_epochs)

    with SessionWithExitSave(saver=model.train_saver, exit_save_path='output/tf-saves/lastest.ckpt',
                             graph=graph) as sess:
        print("\n[Network/Training] Initializing graph's variables...")

        if args.retrain:
            print('\n[network/Training] Loading the model to retrain it...')
            load_model(saver=model.train_saver, sess=sess)
        else:
           init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
           sess.run(init_op)

        # ===============
        #  Training Loop
        # ===============
        print("[Network/Training] Training Initialized!\n")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        timer = -time.time()
        for step in range(1, args.max_steps + 1):
            # --------------------- #
            # [Train] Session Run!  #
            # --------------------- #
            timer2 = -time.time()

            _, \
            batch_image, \
            batch_image_key, \
            batch_image_uint8, \
            batch_depth, \
            batch_depth_key, \
            batch_pred, \
            model.train.loss, \
            summary_train_loss = sess.run([model.train_step,
                                           model.train.tf_batch_image,
                                           model.train.tf_batch_image_key,
                                           model.train.tf_batch_image_uint8,
                                           model.train.tf_batch_depth,
                                           model.train.tf_batch_depth_key,
                                           model.train.fcrn.get_output(),
                                           model.train.tf_loss,
                                           model.tf_summary_train_loss])

            # Reset network parameters to prevent the model from collapsing
            # TODO: Validar. Se a condição for facilmente/frequentemente atingida, o modelo talvez não convirja nunca.
            # TODO: Adicionar um contador para evitar falsos positivos
            # TODO: Adicionar contador, caso o master reset for acionado mais que N vezes. Abortar treinamento.
            if np.max(batch_pred) < 0.2:
                print("[Train] MASTER RESET triggered!!! max(batch_pred):", np.max(batch_pred))
                sess.run(init_op)

            # # Detect Invalid Pairs
            # for i in range(args.batch_size):
            #     print(i, batch_image_key[i], batch_depth_key[i])
            #     image_head, image_tail = os.path.split(batch_image_key[i].decode("utf-8"))
            #     depth_head, depth_tail = os.path.split(batch_depth_key[i].decode("utf-8"))
            #
            #     if image_tail.split('_')[0] != depth_tail.split('_')[0]:
            #         input("Invalid Pair Detected!")
            # print()

            model.summary_writer.add_summary(summary_train_loss, step)

            # Prints Training Progress
            if step % 10 == 0:
                if args.show_train_progress:
                    model.train.plot.show_train_results(raw=batch_image_uint8[0],
                                                        label=batch_depth[0, :, :, 0],
                                                        pred=batch_pred[0, :, :, 0])

                timer2 += time.time()

                print(
                    'epoch: {0:d}/{1:d} | step: {2:d}/{3:d} | t: {4:f} | Batch trLoss: {5:>16.4f} | vLoss: {6:>16.4f} '.format(
                        epoch,
                        max_epochs,
                        step,
                        args.max_steps,
                        timer2,
                        model.train.loss,
                        model.valid.loss))

            if step % 1000 == 0:
                model.summary_writer.flush()  # Don't forget this command! It makes sure Python writes the summaries to the log-file

            # -------------------------- #
            # [Validation] Session Run!  #
            # -------------------------- #
            # [Valid] TODO: Portar Leitura para o Tensorflow
            # [Valid] TODO: Implementar Leitura por Batches

            # Detects the end of a epoch
            # if True: # Only for testing the following condition!!!
            # if (np.floor((step * args.batch_size) / data.numTrainSamples) != epoch) and not TRAIN_ON_SINGLE_IMAGE:
            if step % 1000 == 0 and not TRAIN_ON_SINGLE_IMAGE:
                valid_loss_sum = 0
                print("\n[Network/Validation] Epoch finished. Starting TestData evaluation...")
                for i in range(data.num_test_samples):
                    timer3 = -time.time()
                    feed_valid = {model.valid.tf_image_key: data.test_image_filenames[i],
                                  model.valid.tf_depth_key: data.test_depth_filenames[i]}

                    # valid_image_key, valid_depth_key = sess.run([model.valid.tf_image_key, model.valid.tf_depth_key], feed_valid)
                    # print(valid_image_key, valid_depth_key)

                    valid_image, \
                    valid_image_uint8, \
                    valid_pred, \
                    valid_depth, \
                    model.valid.loss = sess.run([model.valid.tf_image_resized,
                                                 model.valid.tf_image_resized_uint8,
                                                 model.valid.tf_pred,
                                                 model.valid.tf_depth_resized,
                                                 model.valid.tf_loss],
                                                feed_dict=feed_valid)

                    if args.show_valid_progress:
                        model.valid.plot.show_train_results(raw=valid_image_uint8,
                                                            label=valid_depth[:, :, 0],
                                                            pred=valid_pred[0, :, :, 0])

                    valid_loss_sum += model.valid.loss

                    timer3 += time.time()
                    print("%d/%d | valid_loss_sum: %f | valid_loss: %f | t: %4f" % (
                        i + 1, data.num_test_samples, valid_loss_sum, model.valid.loss, timer3))

                # Calculate mean value of 'valid_loss'
                model.valid.loss = valid_loss_sum / data.num_test_samples  # Updates 'Valid_loss' value
                print("mean(valid_loss): %f\n" % model.valid.loss)

                if ENABLE_EARLY_STOP and model.train.stop.check(step, model.valid.loss):  # TODO: Validar
                    break

                # Write information to TensorBoard
                if ENABLE_TENSORBOARD:
                    summary_str = sess.run(model.summary_op, feed_valid)
                    model.summary_writer.add_summary(summary_str, step)
                    model.summary_writer.flush()  # Don't forget this command! It makes sure Python writes the summaries to the log-file

            epoch = int(np.floor((step * args.batch_size) / data.num_train_samples))

        # End of Training!
        coord.request_stop()
        coord.join(threads)

        timer += time.time()

        print("\n[Network/Training] Training FINISHED! Time elapsed: %f s\n" % timer)

        # ==============
        #  Save Results
        # ==============
        model.save_trained_model(settings.get_save_restore_path(), sess, model.train_saver, args.model_name)
        model.save_results(settings.datetime, epoch, max_epochs, step, args.max_steps, timer)

        sess.close()


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
                raise SystemError

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


# ======
#  Main
# ======
def main():
    print('\n[{}] Selected Params: \n\n{}\n'.format(settings.appName, args))
    print('[{}] Selected mode: {}'.format(settings.appName, args.mode.capitalize()))

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    elif args.mode == 'pred':
        predict()
    else:
        print("[ModeError] Selected mode doesn't exist! Select one of the following: 'train', 'test', or 'pred'.")
        raise SystemExit

    print("\n[%s] Done." % settings.appName)
    sys.exit()


if __name__ == '__main__':
    tf.app.run(main=main())
