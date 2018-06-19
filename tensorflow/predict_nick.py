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
# [Train] FIXME: -v option só funciona se a opção -t também estiver ativada

# [Test] TODO: Validar Métricas
# [Test] TODO: Realizar Tests comparando KittiDepth x KittiDiscrete (disp1) x KittiContinuous (disp2)

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
import warnings
import time
import pyxhook
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imageio
import sys

from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

# Custom Libraries
import modules.args as argsLib
import modules.metrics as metricsLib

from modules.dataloader import Dataloader
from modules.framework import Model
from modules.model.fcrn import ResNet50UpProj
from modules.size import Size
from modules.plot import Plot
from modules.utils import total_size

# ==========================
#  [Train] Framework Config
# ==========================
# Select to consider only the valid Pixels (True) OR ALL Pixels (False)
VALID_PIXELS = False             # Default: True

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


# ===========
#  Functions
# ===========
def getSaveFolderPaths():
    """Defines folders paths for saving the model variables to disk."""
    valid_px_str = 'valid_px' if VALID_PIXELS else 'all_px'
    relative_save_path = 'output/' + appName + '/' + args.dataset + '/' + valid_px_str + '/' + args.loss + '/' + datetime + '/'
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
            data.train_image_filenames = tf.expand_dims(data.train_image_filenames[0], axis=0)
            data.train_depth_filenames = tf.expand_dims(data.train_depth_filenames[0], axis=0)

        data.tf_train_image, data.tf_train_depth = data.readData(data.train_image_filenames, data.train_depth_filenames)

        # Build Network Model
        model = Model(args, data, args.loss, VALID_PIXELS)
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
        for step in range(args.max_steps + 1):
            if running:
                # --------------------- #
                # [Train] Session Run!  #
                # --------------------- #
                timer2 = -time.time()

                _, \
                batch_data, \
                batch_data_uint8, \
                batch_labels, \
                batch_pred, \
                model.train.loss, \
                summary_train_loss = sess.run([model.train_step,
                                               model.train.tf_batch_data,
                                               model.train.tf_batch_data_uint8,
                                               model.train.tf_batch_labels,
                                               model.train.fcrn.get_output(),
                                               model.train.tf_loss,
                                               model.tf_summary_train_loss])

                model.summary_writer.add_summary(summary_train_loss, step)

                def debug_data_augmentation():
                    fig, axes = plt.subplots(nrows=2, ncols=2)

                    axes[0, 0].set_title('images_resized')
                    axes[0, 0].imshow(images_resized)

                    axes[0, 1].set_title('depths_resized[:, :, 0]')
                    axes[0, 1].imshow(depths_resized[:, :, 0])

                    axes[1, 0].set_title('images_proc')
                    axes[1, 0].imshow(images_proc)

                    axes[1, 1].set_title('depths_proc[:,:,0]')
                    axes[1, 1].imshow(depths_proc[:, :, 0])
                    fig.tight_layout()

                    plt.pause(0.001)
                    input("proc")

                # debug_data_augmentation() # TODO: Terminar

                # Prints Training Progress
                if step % 10 == 0:
                    if args.show_train_progress:
                        # plt.figure(100)
                        # plt.imshow(pred2[0, :, :, 0])

                        model.train.plot.showResults(raw=batch_data_uint8[0],
                                                     label=batch_labels[0, :, :, 0],
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
                if (np.floor((step * args.batch_size) / data.numTrainSamples) != epoch) and not TRAIN_ON_SINGLE_IMAGE:
                    valid_loss_sum = 0
                    print("\n[Network/Validation] Epoch finished. Starting TestData evaluation...")
                    for i in range(data.numTestSamples):
                        timer3 = -time.time()
                        # TODO: Otimizar
                        valid_image = imageio.imread(data.test_image_filenames[i])
                        valid_depth = imageio.imread(data.test_depth_filenames[i])
                        feed_valid = {model.valid.tf_image: np.expand_dims(valid_image, axis=0),
                                      model.valid.tf_depth: np.expand_dims(np.expand_dims(valid_depth, axis=0), axis=3)}

                        valid_image, \
                        valid_image_uint8, \
                        valid_pred, \
                        valid_labels, \
                        model.valid.loss = sess.run([model.valid.tf_image_resized,
                                                     model.valid.tf_image_resized_uint8,
                                                     model.valid.tf_pred,
                                                     model.valid.tf_depth_resized,
                                                     model.valid.tf_loss],
                                                    feed_dict=feed_valid)

                        if args.show_valid_progress:
                            model.valid.plot.showResults(raw=valid_image_uint8[0, :, :],
                                                         label=valid_labels[0, :, :, 0],
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
                        summary = sess.run(model.summary_op, feed_valid)
                        model.summary_writer.add_summary(summary, step)
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

    # Construct the network
    with tf.variable_scope('model'):
        input_size = Size(228, 304, 3)
        output_size = Size(128, 160, 1)
        batch_size = 1

        tf_image_path = tf.placeholder(tf.string)
        tf_depth_path = tf.placeholder(tf.string)

        if data.dataset_name == 'apolloscape':
            tf_image = tf.image.decode_jpeg(tf.read_file(tf_image_path), channels=3)
        else:
            tf_image = tf.image.decode_png(tf.read_file(tf_image_path), channels=3, dtype=tf.uint8)

        if data.dataset_name.split('_')[0] == 'kittidiscrete' or \
           data.dataset_name.split('_')[0] == 'kitticontinuous':
            tf_depth = tf.image.decode_png(tf.read_file(tf_depth_path), channels=1, dtype=tf.uint8)
        else:
            tf_depth = tf.image.decode_png(tf.read_file(tf_depth_path), channels=1, dtype=tf.uint16)

        # TODO: Remover? Segundo o vitor não faz sentido remover o céu no test
        removeSky = True
        if removeSky:
            # Crops Input and Depth Images (Removes Sky)
            if data.dataset_name[0:5] == 'kitti':
                tf_image_shape = tf.shape(tf_image)
                tf_depth_shape = tf.shape(tf_depth)

                crop_height_perc = tf.constant(0.3, tf.float32)
                tf_image_new_height = crop_height_perc * tf.cast(tf_image_shape[0], tf.float32)
                tf_depth_new_height = crop_height_perc * tf.cast(tf_depth_shape[0], tf.float32)

                tf_image = tf_image[tf.cast(tf_image_new_height, tf.int32):, :]
                tf_depth = tf_depth[tf.cast(tf_depth_new_height, tf.int32):, :]

        # True Depth Value Calculation. May vary from dataset to dataset.
        tf_depth = data.rawdepth2meters(tf_depth)

        # tf_image.set_shape(input_size.getSize())
        # tf_depth.set_shape(output_size.getSize())

        # Downsizes Input and Depth Images
        tf_image_resized = tf.image.resize_images(tf_image, [input_size.height, input_size.width])
        tf_image_resized_uint8 = tf.cast(tf_image_resized, tf.uint8)  # Visual purpose
        tf_image_resized = tf.expand_dims(tf_image_resized, axis=0)  # Model's Input size requirement

        tf_depth_resized = tf.image.resize_images(tf_depth, [output_size.height, output_size.width])

        net = ResNet50UpProj({'data': tf_image_resized}, batch=batch_size, keep_prob=1, is_training=False)
        tf_pred = net.get_output()

        tf_pred_up = tf.image.resize_images(tf_pred, tf.shape(tf_depth)[:2], tf.image.ResizeMethod.BILINEAR, False)

        # Group Tensors
        image_op = [tf_image_path, tf_image, tf_image_resized_uint8]
        depth_op = [tf_depth_path, tf_depth, tf_depth_resized]
        pred_op = [tf_pred, tf_pred_up]

        # Print Tensors
        print("\nTensors:")
        print(tf_image_path)
        print(tf_depth_path)
        print(tf_image)
        print(tf_depth)
        print(tf_image_resized)
        print(tf_image_resized_uint8)
        print(tf_pred)
        print(tf_pred_up)

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
                feed_test = {tf_image_path: data.test_image_filenames[i], tf_depth_path: data.test_depth_filenames[i]}

                _, image, image_resized = sess.run(image_op, feed_test)
                _, depth, depth_resized = sess.run(depth_op, feed_test)
                pred, pred_up = sess.run(pred_op, feed_test)
            else:
                feed_test = {tf_image_path: data.test_image_filenames[i]}
                _, image, image_resized = sess.run(image_op, feed_test)
                pred, pred_up = sess.run(pred_op, feed_test)

            pred_list.append(pred_up[0])
            gt_list.append(depth)

            # print(image.shape)
            # print(image_resized.shape)
            # print(depth.shape)
            # print(depth_resized.shape)
            # print(pred.shape)
            # input("test")

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
        # if data.test_depth_filenames:
        #     pred_array = np.array(pred_list)
        #     gt_array = np.array(gt_list)
        #
        #     def evaluateTestSet(pred, gt, mask):
        #         # Compute error metrics on benchmark datasets
        #         # -------------------------------------------------------------------------
        #
        #         # make sure predictions and ground truth have same dimensions
        #         if pred.shape != gt_array.shape:
        #             # pred = imresize(pred, [size(gt, 1), size(gt, 2)], 'bilinear') # TODO: Terminar
        #             input("terminar!")
        #             pass
        #
        #         if mask is None:
        #             n_pxls = gt.size
        #         else:
        #             n_pxls = len(gt[mask])  # average over valid pixels only # TODO: Terminar
        #
        #         print('\n Errors computed over the entire test set \n')
        #         print('------------------------------------------\n')
        #
        #         # Mean Absolute Relative Error
        #         rel = np.abs(gt - pred)/ gt  # compute errors
        #
        #         print(pred.shape, pred.size)
        #         print(gt.shape, gt.size)
        #         print(n_pxls)
        #         print(rel)
        #         print(rel[mask])
        #
        #         print(rel)
        #         input("antes")
        #         rel[mask] = 0
        #         print(rel)
        #         input("depois")
        #
        #         # rel(~mask) = 0                      # mask out invalid ground truth pixels
        #         # rel = sum(rel) / n_pxls             # average over all pixels
        #         # print('Mean Absolute Relative Error: %4f\n', rel)
        #         #
        #         # # Root Mean Squared Error
        #         # rms = (gt - pred)**2
        #         # rms(~mask) = 0
        #         # rms = sqrt(sum(rms) / n_pxls)
        #         # print('Root Mean Squared Error: %4f\n', rms)
        #         #
        #         # # LOG10 Error
        #         # lg10 = abs(log10(gt) - log10(pred))
        #         # lg10(~mask) = 0
        #         # lg10 = sum(lg10) / n_pxls
        #         # print('Mean Log10 Error: %4f\n', lg10)
        #         #
        #         # results.rel = rel
        #         # results.rms = rms
        #         # results.log10 = lg10
        #
        #         return results
        #
        #     if VALID_PIXELS:
        #         mask = np.where(gt_array > 0) # TODO: Adicionar ranges para cada um dos datasets
        #         # print(len(mask))
        #
        #         imask = tf.where(gt_array > 0, tf.ones_like(gt_array), tf.zeros_like(depth))
        #         depth2 = tf_depth * tf_imask
        #
        #     else:
        #         mask = None
        #
        #     evaluateTestSet(pred_array, gt_array, mask)
        #     # metricsLib.evaluateTesting(pred, test_labels_o)

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
