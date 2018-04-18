#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ============
#  To-Do FCRN
# ============
# TODO: Implementar leitura das imagens pelo Tensorflow - Teste
# TODO: Validar Métricas.

# TODO: Implementar Bilinear
# TODO: If detect Ctrl+C, save training state.
# TODO: Estou aplicando a normalização da entrada em todos os módulos (predict, test, train, valid)?

# ===========
#  Libraries
# ===========
import os
import warnings
import time
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import utils.metrics as metricsLib
import utils.args as argsLib

from PIL import Image
from utils.dataloader import Dataloader
from utils.model import Model
from utils.fcrn import ResNet50UpProj
from utils.train import EarlyStopping
from utils.size import Size
from utils.plot import Plot

# ==================
#  Global Variables
# ==================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")  # Suppress Warnings

appName = 'fcrn'
datetime = time.strftime("%Y-%m-%d") + '_' + time.strftime("%H-%M-%S")

# Select the Loss Function:
# 0 - MSE
# 1 - Eigen's Log Depth
# 2 - BerHu
LOSS_FUNCTION = 0

# Select to consider only the valid Pixels (True) OR ALL Pixels (False)
VALID_PIXELS = True  # Default: True
TRAIN_ON_SINGLE_IMAGE = True  # Default: False
ENABLE_EARLY_STOP = False  # Default: True # TODO: Ativar
SAVE_TRAINED_MODEL = True  # Default: True
ENABLE_TENSORBOARD = True  # Default: True
SAVE_TEST_DISPARITIES = True  # Default: True
APPLY_BILINEAR_OUTPUT = False  # Default: False

LOG_INITIAL_VALUE = 1
# ===========
#  Functions
# ===========
def createSaveFolder():
    save_path = None
    save_restore_path = None

    if SAVE_TRAINED_MODEL or ENABLE_TENSORBOARD:
        # Saves the model variables to disk.
        relative_save_path = 'output/' + appName + '/' + datetime + '/'
        save_path = os.path.join(os.getcwd(), relative_save_path)
        save_restore_path = os.path.join(save_path, 'restore/')

        if not os.path.exists(save_restore_path):
            os.makedirs(save_restore_path)

    return save_path, save_restore_path


# ========= #
#  Predict  #
# ========= #
def predict(model_data_path, image_path):
    print('[%s] Selected mode: Predict' % appName)
    print('[%s] Selected Params: \n\n%s\n' % (appName, args))

    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    # Read image
    img = Image.open(image_path)
    img = img.resize([width, height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis=0)

    # Create a placeholder for the input image
    tf_image = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    with tf.variable_scope('model'):
        # Construct the network
        net = ResNet50UpProj({'data': tf_image}, batch_size, 1, False)

        # for var in tf.trainable_variables():
        #     print(var)

    with tf.Session() as sess:
        # Load the converted parameters
        print('\n[network/Predict] Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        # net.load(model_data_path, sess)

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={tf_image: img})

        # Plot result
        fig = plt.figure()
        ii = plt.imshow(pred[0, :, :, 0], interpolation='nearest')
        fig.colorbar(ii)
        plt.show()

        return pred


# ===================== #
#  Training/Validation  #
# ===================== #
def train(args):
    print('[%s] Selected mode: Train' % appName)
    print('[%s] Selected Params: \n\n%s\n' % (appName, args))

    # Local Variables
    save_path, save_restore_path = createSaveFolder()  # TODO: Evitar criar pastas vazias

    # ----------------------------------------- #
    #  Network Training Model - Building Graph  #
    # ----------------------------------------- #
    graph = tf.Graph()
    with graph.as_default():
        data = Dataloader(args)
        model = Model(args)

        # Searches dataset images filenames
        image_filenames, depth_filenames, tf_image_filenames, tf_depth_filenames = data.getTrainData(args)

        # TODO: mudar nome das variaveis para algo do tipo dataset.train.image_filenames e dataset.train.depth_filenames
        data.splitData(image_filenames, depth_filenames)

        # If enabled, the framework will train the network for only one image!!!
        if TRAIN_ON_SINGLE_IMAGE:
            data.train_image_filenames = tf.expand_dims(data.train_image_filenames[0], axis=0)
            data.train_depth_filenames = tf.expand_dims(data.train_depth_filenames[0], axis=0)

        tf_train_image, tf_train_depth = data.readData(data.train_image_filenames, data.train_depth_filenames)

        # Build Network Model
        model.build_model(data.image_size, data.depth_size, tf_train_image, tf_train_depth)
        model.build_losses(LOSS_FUNCTION, VALID_PIXELS)
        model.build_optimizer()
        model.build_summaries()
        model.countParams()

        model.collectSummaries(save_path, graph)
        model.createTrainSaver()

    # ---------------------------------------- #
    #  Network Training Model - Running Graph  #
    # ---------------------------------------- #
    # Local Variables and Memory Allocation
    step = 0
    stop = EarlyStopping()

    with tf.Session(graph=graph) as sess:
        print("\n[Network/Training] Initializing graph's variables...")
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        # Check Dataset Integrity
        data.checkIntegrity(sess, tf_image_filenames, tf_depth_filenames)

        # Proclaim the epochs
        epochs = np.floor(args.batch_size * args.max_steps / data.numSamples)
        print('\nTrain with approximately %d epochs' % epochs)

        # ===============
        #  Training Loop
        # ===============
        if args.show_train_progress:
            train_plotObj = Plot(args.mode, title='Train Predictions')

        if args.show_valid_progress:
            valid_plotObj = Plot(args.mode, title='Validation Prediction')

        print("[Network/Training] Training Initialized!\n")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        start = time.time()
        for step in range(args.max_steps):
            start2 = time.time()

            # ----- Session Run! ----- #
            # Training
            _, batch_data_raw, batch_data, batch_labels, log_batch_labels, batch_pred, model.train.loss, summary_str = sess.run(
                [model.train_step, model.train.tf_batch_data_resized, model.train.tf_batch_data,
                 model.train.tf_batch_labels, model.train.tf_log_batch_labels,
                 model.fcrn.get_output(), model.train.tf_loss, model.summary_op])

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

            # debug_data_augmentation()

            # Validation
            # FIXME: Uses only one image as validation!
            # FIXME: valid_loss value may is wrong
            feed_valid = {model.valid.tf_image: np.expand_dims(plt.imread(data.valid_image_filenames[0]), axis=0),
                          model.valid.tf_depth: np.expand_dims(
                              np.expand_dims(plt.imread(data.valid_depth_filenames[0]), axis=0), axis=3)}
            valid_image, valid_pred, valid_labels, valid_log_labels, model.valid.loss = sess.run(
                [model.valid.tf_image_resized, model.fcrn_valid.get_output(), model.valid.tf_depth_resized,
                 model.valid.tf_log_depth_resized, model.valid.tf_loss], feed_dict=feed_valid)
            # -----

            if ENABLE_TENSORBOARD:
                # Write information to TensorBoard
                model.summary_writer.add_summary(summary_str, step)
                model.summary_writer.flush()  # Don't forget this command! It makes sure Python writes the summaries to the log-file

            # TODO: Validar
            if ENABLE_EARLY_STOP:
                if stop.check(step, model.valid.loss):
                    break

            # Prints Training Progress
            if step % 10 == 0:
                if args.show_train_progress:
                    train_plotObj.showResults(raw=batch_data_raw[0],
                                              label=batch_labels[0, :, :, 0],
                                              log_label=log_batch_labels[0, :, :, 0],
                                              pred=batch_pred[0, :, :, 0],
                                              cbar_range=data.datasetObj)

                if args.show_valid_progress:
                    valid_plotObj.showResults(raw=valid_image[0, :, :],
                                              label=valid_labels[0, :, :, 0],
                                              log_label=valid_log_labels[0, :, :, 0],
                                              pred=valid_pred[0, :, :, 0],
                                              cbar_range=data.datasetObj)

                end2 = time.time()
                print('step: {0:d}/{1:d} | t: {2:f} | Batch trLoss: {3:>16.4f} | vLoss: {4:>16.4f} '.format(step,
                                                                                                            args.max_steps,
                                                                                                            end2 - start2,
                                                                                                            model.train.loss,
                                                                                                            model.valid.loss))

        coord.request_stop()
        coord.join(threads)

        end = time.time()
        sim_train = end - start
        print("\n[Network/Training] Training FINISHED! Time elapsed: %f s\n" % sim_train)

        # ==============
        #  Save Results
        # ==============
        if SAVE_TRAINED_MODEL:
            model.saveTrainedModel(save_restore_path, sess, model.train_saver, args.model_name)

        # Logs the obtained simulation results
        print("[Results] Logging simulation info to 'results.txt' file...")
        f = open('results.txt', 'a')
        f.write("%s\t\t%s\t\t%s\t\t%s\t\tsteps: %d\ttrain_loss: %f\tvalid_loss: %f\tt: %f s\n" % (
            datetime, args.model_name, args.dataset, model.loss_name, step, model.train.loss, model.valid.loss,
            sim_train))
        f.close()


# ========= #
#  Testing  #
# ========= #
def test(args):
    print('[%s] Selected mode: Test' % appName)
    print('[%s] Selected Params: \n\n%s\n' % (appName, args))

    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    # -----------------------------------------
    #  Network Testing Model - Importing Graph
    # -----------------------------------------
    # Loads the dataset and restores a specified trained model.
    data = Dataloader(args)  # TODO: Usar leitura pelo Tensorflow

    # Searches dataset images filenames
    test_image_filenames, test_depth_filenames, tf_test_image_filenames, tf_test_depth_filenames = data.getTestData(args)

    # Create a placeholder for the input image
    tf_image = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    with tf.variable_scope('model'):
        # Construct the network
        net = ResNet50UpProj({'data': tf_image}, batch_size, 1, False)
        input_size = Size(228, 304, 3)
        output_size = Size(128, 160, 1)

    # Memory Allocation
    # Length of test_dataset used, so when there is not test_labels, the variable will still be declared.
    test_data_o = np.zeros((data.numSamples, input_size.height, input_size.width, input_size.nchannels), dtype=np.uint8)  # (?, 172, 576, 3)
    test_data_crop_o = np.zeros((data.numSamples, input_size.height, input_size.width, input_size.nchannels), dtype=np.uint8)  # (?, 172, 576, 3)
    pred = np.zeros((data.numSamples, output_size.height, output_size.width), dtype=np.float32)  # (?, 43, 144)
    test_labels_o = np.zeros((data.numSamples, output_size.height, output_size.width), dtype=np.int32)  # (?, 43, 144)

    with tf.Session() as sess:
        # Load the converted parameters
        print('\n[network/Testing] Loading the model...')

        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, args.model_path)

        # Use to load from npy file
        # net.load(model_data_path, sess)

        # ==============
        #  Testing Loop
        # ==============
        start = time.time()

        for i, image_path in enumerate(test_image_filenames):
            start2 = time.time()

            if test_depth_filenames:  # It's not empty
                image, depth, image_crop, depth_bilinear = data.readImage(test_image_filenames[i],
                                                                          test_depth_filenames[i],
                                                                          input_size,
                                                                          output_size,
                                                                          mode='test')

                test_labels_o[i] = depth[:, :, 0]
                # test_labelsBilinear_o[i] = depth_bilinear # TODO: Usar?
            else:
                image, _, image_crop, _ = data.readImage(data.test_dataset[i], None, mode='test')

            test_data_o[i] = image
            test_data_crop_o[i] = image_crop

            # Evalute the network for the given image
            pred_temp = sess.run(net.get_output(), feed_dict={tf_image: np.expand_dims(np.asarray(test_data_o[i]), axis=0)})
            pred[i] = pred_temp[:, :, :, 0]

            # Prints Testing Progress
            end2 = time.time()
            print('step: %d/%d | t: %f' % (i + 1, data.numSamples, end2 - start2))
            # break # Test

        # Testing Finished.
        end = time.time()
        print("\n[Network/Testing] Testing FINISHED! Time elapsed: %f s" % (end - start))

        # ==============
        #  Save Results
        # ==============
        # Saves the Test Predictions
        print("[Network/Testing] Saving testing predictions...")

        output_directory = os.path.dirname(args.model_path) if args.output_directory == '' else args.output_directory

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        if SAVE_TEST_DISPARITIES:
            np.save(output_directory[:-7] + 'test_pred.npy', pred)  # The indexing removes 'restore' from folder path

        # Calculate Metrics
        if test_depth_filenames:
            metricsLib.evaluateTesting(pred, test_labels_o)
        else:
            print(
                "[Network/Testing] It's not possible to calculate Metrics. There are no corresponding labels for Testing Predictions!")

        # Show Results
        if args.show_test_results:
            test_plotObj = Plot(args.mode, title='Test Predictions')
            for i in range(data.numSamples):
                test_plotObj.showTestResults(raw=test_data_crop_o[i],
                                             label=test_labels_o[i],
                                             log_label=np.log(test_labels_o[i] + LOG_INITIAL_VALUE),
                                             pred=pred[i], i=i)


# ======
#  Main
# ======
def main(args):
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'pred':
        predict(args.model_path, args.image_path)

    print("\n[%s] Done." % appName)
    sys.exit()


# ======
#  Main
# ======
if __name__ == '__main__':
    args = argsLib.argumentHandler()

    # Limits Tensorflow to see only the specified GPU.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    tf.app.run(main=main(args))
