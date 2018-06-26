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

# [Train] FIXME: Early Stopping

# [Test] TODO: Procurar métricas mais recentes de outros trabalhos
# [Test] TODO: Ver métricas do trabalho DORN. Dep: Instalar Caffe
# [Test] TODO: Ver métricas do Kitti para Depth Estimation
# [Test] TODO: Realizar Tests comparando KittiDepth x KittiDiscrete (disp1) x KittiContinuous (disp2)
# [Test] TODO: Implementar Métricas em Batches
# [Test] TODO: Validar Métricas

# Known Bugs
# [Train][Major Bug!!!] FIXME: Os pares de treinamento ficam desalinhados. Já havia detectado este problema. O problema abaixo pode estar relacionado
# TODO: Por que string_input_producer sempre começa do segundo sample?
# As vezes a leitura das strings ficam desalinhas, já havia detectado este problema anteriormente
# [Train] FIXME: O que causa aquelas predições com pixeis de intensidade alta? Devo ou não clippar as predições?
# [Train] FIXME: Arrumar outras transformações de Data Augmentation, atualmente apenas a transformação de flip está funcionando

# Optional
# [Dataset] FIXME: Descobrir porquê o código do vitor (cnn_hilbert) não está gerando todas as imagens (disp1 e disp2)
# [Train] TODO: Dar suporte ao Make3D
# [Train] TODO: Adicionar feature para realizar pré-carregamento do modelo pré-treinado no ImageNet

# Ideas
# TODO: Trabalhar com Sequências Temporais: Semelhante à SfM, LSTM
# TODO: Como Monocular Depth pode auxiliar em Visual Odometry?
# TODO: O trabalho "Sparsity Invariant CNNs" diz que redes neurais devem ser capazes de distinguir pixeis observados e pixeis inválidos. Não simplesmente "mask them out".
# TODO: Investigar Redes Neurais que estudam esparsidade DENTRO das redes e nas ENTRADAS. Ref: "Sparsity Invariant CNNs"

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
import pyxhook
import tensorflow as tf
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Custom Libraries
import modules.args as argsLib
import modules.metrics as myMetrics
import modules.metrics_laina as LainaMetrics
import modules.metrics_monodepth as MonodepthMetrics
from modules.dataloader import Dataloader
from modules.framework import Model
from modules.model.fcrn import ResNet50UpProj
from modules.plot import Plot
from modules.test import Test
from modules.utils import total_size

# ==========================
#  [Train] Framework Config
# ==========================
TRAIN_ON_SINGLE_IMAGE = False   # Default: False
ENABLE_EARLY_STOP = True        # Default: True
ENABLE_TENSORBOARD = True       # Default: True
SAVE_TRAINED_MODEL = True       # Default: True

# =========================
#  [Test] Framework Config
# =========================
# Select Subset:
# 0 - TestData                  # Default
# 1 - TrainData
TEST_EVALUATE_SUBSET = 0

SAVE_TEST_DISPARITIES = True    # Default: True

# ==================
#  Global Variables
# ==================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")  # Suppress Warnings

appName = 'fcrn'
datetime = time.strftime("%Y-%m-%d") + '_' + time.strftime("%H-%M-%S")
LOG_INITIAL_VALUE = 1


# ===========
#  Functions
# ===========
def getSaveFolderPaths():
    """Defines folders paths for saving the model variables to disk."""
    px_str = args.px + '_px'
    relative_save_path = 'output/' + appName + '/' + args.dataset + '/' + px_str + '/' + args.loss + '/' + datetime + '/'
    save_path = os.path.join(os.getcwd(), relative_save_path)
    save_restore_path = os.path.join(save_path, 'restore/')

    return save_path, save_restore_path


# TODO: Move
# This function is called every time a key is presssed
def kbevent(event):
    # print key info
    # print(event)

    # If the ascii value matches spacebar, terminate the while loop
    if event.Ascii == 197:  # Press 'F8' to stop training.
        global running
        running = False


# Create hookmanager
hookman = pyxhook.HookManager()
# Define our callback to fire when a key is pressed down
hookman.KeyDown = kbevent
# Hook the keyboard
hookman.HookKeyboard()
# Start our listener
hookman.start()


# ========= #
#  Predict  #
# ========= #
def predict(model_data_path, image_path):
    print('[%s] Selected mode: Predict' % appName)

    # Default input size
    batch_size, height, width, nchannels = 1, 228, 304, 3

    # Read image (uint8)
    img = Image.open(image_path)
    img = np.array(img)

    # ------- #
    #  Graph  #
    # ------- #
    # Create a placeholder for the input image
    tf_image = tf.placeholder(tf.uint8, shape=(None, None, 3))
    tf_image_resized = tf.image.resize_images(tf_image, [height, width])
    tf_image_resized_uint8 = tf.cast(tf_image_resized, tf.uint8)  # Visual purpose
    tf_image_input = tf.expand_dims(tf_image_resized, axis=0)

    with tf.variable_scope('model'):
        # Construct the network
        net = ResNet50UpProj({'data': tf_image_input}, batch=batch_size, keep_prob=1, is_training=False)
        tf_pred = net.get_output()
        # for var in tf.trainable_variables():
        #     print(var)

    # Merge Ops
    pred_op = [tf_image, tf_image_resized_uint8, tf_image_input, tf_pred]

    # Print Variables
    # print(img)
    print(img.shape, img.dtype)

    print(tf_image)
    print(tf_image_resized)
    print(tf_image_input)
    print(tf_pred)

    with tf.Session() as sess:
        # Load the converted parameters
        print('\n[network/Predict] Loading the model')

        # --------- #
        #  Restore  #
        # --------- #
        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        # net.load(model_data_path, sess)

        # ----- #
        #  Run  #
        # ----- #
        # Evalute the network for the given image
        image, image_resized_uint8, image_input, pred = sess.run(pred_op, feed_dict={tf_image: img})

        # --------- #
        #  Results  #
        # --------- #
        fig = plt.figure(figsize=(15, 5))
        fig.subplots_adjust(bottom=0.025, left=0.025, top=0.975, right=0.975)
        X = [(1, 5, (1, 2)), (1, 5, 3), (1, 5, 4), (1, 5, 5)]
        axes = []
        for nrows, ncols, plot_number in X:
            axes.append(fig.add_subplot(nrows, ncols, plot_number))

        img1 = axes[0].imshow(image)
        img2 = axes[1].imshow(image_resized_uint8)
        img3 = axes[2].imshow(image_input[0])
        img4 = axes[3].imshow(pred[0, :, :, 0])
        # img4 = axes[3].imshow(pred[0, :, :, 0], interpolation='nearest')

        axes[0].set_title('Image')
        axes[1].set_title('Resized')
        axes[2].set_title('Input')
        axes[3].set_title('Pred')

        # Fix Colorbar size
        divider = make_axes_locatable(axes[3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img4, cax=cax)

        plt.show()

        return pred


# ===================== #
#  Training/Validation  #
# ===================== #
def train(args):
    print('[%s] Selected mode: Train' % appName)

    # Local Variables
    global running  # Create a loop to keep the application running
    running = True

    save_path, save_restore_path = getSaveFolderPaths()

    # ----------------------------------------- #
    #  Network Training Model - Building Graph  #
    # ----------------------------------------- #
    graph = tf.Graph()
    with graph.as_default():
        data = Dataloader(args)

        # If enabled, the framework will train the network for only one image!!!
        if TRAIN_ON_SINGLE_IMAGE:
            data.train_image_filenames = np.expand_dims(data.train_image_filenames[0], axis=0)
            data.train_depth_filenames = np.expand_dims(data.train_depth_filenames[0], axis=0)

        data.tf_train_image_key, data.tf_train_image, data.tf_train_depth_key, data.tf_train_depth = data.readData(data.train_image_filenames, data.train_depth_filenames)

        # Build Network Model
        model = Model(args, data)
        model.collectSummaries(save_path, graph)
        model.createTrainSaver()

    # ---------------------------------------- #
    #  Network Training Model - Running Graph  #
    # ---------------------------------------- #
    # Local Variables and Memory Allocation
    epoch, step = 0, 0

    # Proclaim the epochs
    max_epochs = int(np.floor(args.batch_size * args.max_steps / data.numTrainSamples))
    print('\nTrain with approximately %d epochs' % max_epochs)

    with tf.Session(graph=graph) as sess:
        print("\n[Network/Training] Initializing graph's variables...")
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
            if running:
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
                        model.train.plot.showResults(raw=batch_image_uint8[0],
                                                     label=batch_depth[0, :, :, 0],
                                                     pred=batch_pred[0, :, :, 0])

                    timer2 += time.time()

                    print('epoch: {0:d}/{1:d} | step: {2:d}/{3:d} | t: {4:f} | Batch trLoss: {5:>16.4f} | vLoss: {6:>16.4f} '.format(
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
                    for i in range(data.numTestSamples):
                        timer3 = -time.time()
                        # TODO: Otimizar
                        valid_image = imageio.imread(data.test_image_filenames[i])
                        valid_depth = imageio.imread(data.test_depth_filenames[i])
                        feed_valid = {model.valid.tf_image_raw: np.expand_dims(valid_image, axis=0),
                                      model.valid.tf_depth_raw: np.expand_dims(np.expand_dims(valid_depth, axis=0), axis=3)}

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
                            model.valid.plot.showResults(raw=valid_image_uint8[0, :, :],
                                                         label=valid_depth[0, :, :, 0],
                                                         pred=valid_pred[0, :, :, 0])

                        valid_loss_sum += model.valid.loss

                        timer3 += time.time()
                        print("%d/%d | valid_loss_sum: %f | valid_loss: %f | t: %4f" % (
                            i + 1, data.numTestSamples, valid_loss_sum, model.valid.loss, timer3))

                    # Calculate mean value of 'valid_loss'
                    model.valid.loss = valid_loss_sum / data.numTestSamples  # Updates 'Valid_loss' value
                    print("mean(valid_loss): %f\n" % model.valid.loss)

                    if ENABLE_EARLY_STOP:
                        if model.train.stop.check(step, model.valid.loss):  # TODO: Validar
                            break

                    # Write information to TensorBoard
                    if ENABLE_TENSORBOARD:
                        summary_str = sess.run(model.summary_op, feed_valid)
                        model.summary_writer.add_summary(summary_str, step)
                        model.summary_writer.flush()  # Don't forget this command! It makes sure Python writes the summaries to the log-file

                epoch = int(np.floor((step * args.batch_size) / data.numTrainSamples))
            else:
                print("[KeyEvent] 'ESC' Pressed! Training process aborted!")
                break

        coord.request_stop()
        coord.join(threads)

        timer += time.time()

        print("\n[Network/Training] Training FINISHED! Time elapsed: %f s\n" % timer)

        # ==============
        #  Save Results
        # ==============
        if SAVE_TRAINED_MODEL:
            if not os.path.exists(save_restore_path):
                os.makedirs(save_restore_path)

            model.saveTrainedModel(save_restore_path, sess, model.train_saver, args.model_name)

        model.saveResults(datetime, epoch, max_epochs, step, args.max_steps, timer)

        sess.close()


# ========= #
#  Testing  #
# ========= #
def test(args):
    print('[%s] Selected mode: Test' % appName)

    # Local Variables
    numSamples = None

    # -----------------------------------------
    #  Network Testing Model - Importing Graph
    # -----------------------------------------
    # Loads the dataset and restores a specified trained model.
    data = Dataloader(args)

    # Searches dataset images filenames
    # data.train_image_filenames, data.train_depth_filenames, tf_train_image_filenames, tf_train_depth_filenames = data.getTrainData()

    if TEST_EVALUATE_SUBSET == 0:
        _ = data.getTestData()
        numSamples = data.numTestSamples
    elif TEST_EVALUATE_SUBSET == 1:
        data.test_image_filenames, data.test_depth_filenames, tf_test_image_filenames, tf_test_depth_filenames = data.getTrainData()
        numSamples = data.numTrainSamples

    model = Test(data)

    with tf.Session() as sess:
        print('\n[network/Testing] Loading the model...')

        # Use to load from *.ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, args.model_path)

        # Use to load from npy file
        # net.load(model_data_path, sess)

        # ==============
        #  Testing Loop
        # ==============
        if args.show_test_results:
            test_plotObj = Plot(args.mode, title='Test Predictions')

        timer = -time.time()
        pred_list, gt_list = [], []
        for i in range(numSamples):
        # for i in range(5): # Only for testing!

            timer2 = -time.time()

            # Evalute the network for the given image
            # data.test_depth_filenames = [] # Only for testing the following condition!!!
            if data.test_depth_filenames:  # It's not empty
                feed_test = {model.tf_image_path: data.test_image_filenames[i],
                             model.tf_depth_path: data.test_depth_filenames[i]}

                _, image, image_resized = sess.run(model.image_op, feed_test)
                _, depth, depth_resized = sess.run(model.depth_op, feed_test)
                pred, pred_up = sess.run(model.pred_op, feed_test)
            else:
                feed_test = {model.tf_image_path: data.test_image_filenames[i]}
                _, image, image_resized = sess.run(model.image_op, feed_test)
                pred, pred_up = sess.run(model.pred_op, feed_test)

            # TODO: Remover, não faz sentido nesta branch (pred depth in meters)
            log_depth = np.log(depth[:, :, 0] + LOG_INITIAL_VALUE)

            # Fill arrays for later on metrics evaluation
            pred_list.append(pred_up[0, :, :, 0])
            gt_list.append(log_depth)

            # Prints Testing Progress
            timer2 += time.time()
            print('step: %d/%d | t: %f | size(pred_list+gt_list): %d' % (i + 1, numSamples, timer2, total_size(pred_list)+total_size(gt_list)))
            # break # Test

            # Show Results
            if args.show_test_results:
                test_plotObj.showTestResults(image=image,
                                             depth=depth[:, :, 0],
                                             image_resized=image_resized,
                                             depth_resized=depth_resized[:, :, 0],
                                             pred=pred[0, :, :, 0],
                                             pred_up=pred_up[0, :, :, 0],
                                             i=i + 1)

        # Testing Finished.
        timer += time.time()
        print("\n[Network/Testing] Testing FINISHED! Time elapsed: %f s" % timer)

        # ==============
        #  Save Results
        # ==============
        # Saves the Test Predictions
        if SAVE_TEST_DISPARITIES:
            print("[Network/Testing] Saving testing predictions...")
            output_directory = os.path.dirname(args.model_path) if args.output_directory == '' else args.output_directory

            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            save_path_pred = os.path.abspath(os.path.join(output_directory, '../')) + '/' + args.dataset + '_pred.npy'

            # TODO: Quais mais variaveis preciso salvar? Não seria melhor salvar a pred_up? Seria legal usar um dictionary?
            # data = {'pred': bla, 'pred_up': bla}
            np.save(save_path_pred, pred)

        # Calculate Metrics
        if data.test_depth_filenames:
            print("[Network/Testing] Calculating Metrics based on Testing Predictions...")

            pred_array = np.array(pred_list)
            gt_array = np.array(gt_list)

            LainaMetrics.evaluate(pred_array, gt_array)
            myMetrics.evaluate(pred_array, gt_array)
            MonodepthMetrics.evaluate(pred_array, gt_array)

        else:
            print("[Network/Testing] It's not possible to calculate Metrics. There are no corresponding labels for Testing Predictions!")


# ======
#  Main
# ======
def main(args):
    print('\n[%s] Selected Params: \n\n%s\n' % (appName, args))

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'pred':
        predict(args.model_path, args.image_path)

    # Close the listener when we are done
    hookman.cancel()

    print("\n[%s] Done." % appName)
    sys.exit()


if __name__ == '__main__':
    args = argsLib.argumentHandler()

    # Limits Tensorflow to see only the specified GPU.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    tf.app.run(main=main(args))
