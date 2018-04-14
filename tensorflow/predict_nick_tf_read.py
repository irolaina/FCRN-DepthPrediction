#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ============
#  To-Do FCRN
# ============
# TODO: Implementar leitura das imagens pelo Tensorflow - Validação
# TODO: Implementar leitura das imagens pelo Tensorflow - Teste
# TODO: Validar Métricas.

# TODO: Implementar Bilinear
# TODO: If detect Ctrl+C, save training state.

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
# from utils.dataloader import Dataloader
from utils.dataloader_new import Dataloader_new  # TODO: make official
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

ENABLE_EARLY_STOP = True
SAVE_TRAINED_MODEL = True
ENABLE_TENSORBOARD = True
SAVE_TEST_DISPARITIES = True
APPLY_BILINEAR_OUTPUT = False


# ===========
#  Functions
# ===========
# TODO: Move
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
    print('[%s] Selected Params: \n\n%s' % (appName, args))

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
    print('[%s] Selected Params: \n\n%s' % (appName, args))

    # Local Variables
    save_path, save_restore_path = createSaveFolder()  # TODO: Evitar criar pastas vazias

    # ----------------------------------------- #
    #  Network Training Model - Building Graph  #
    # ----------------------------------------- #
    graph = tf.Graph()
    with graph.as_default():
        # TODO: Separar algumas imagens para o subset de Validação
        data = Dataloader_new(args)  # TODO: Mudar nome
        model = Model(args)

        # Searches dataset images filenames
        train_image_filenames, train_depth_filenames, tf_train_image_filenames, tf_train_depth_filenames = data.getTrainInputs(
            args)  # TODO: mudar nome das variaveis para algo do tipo dataset.train.image_filenames e dataset.train.depth_filenames
        tf_image, tf_depth = data.readData(tf_train_image_filenames, tf_train_depth_filenames)

        # Downsizes Input and Depth Images
        tf_image_resized = tf.image.resize_images(tf_image, [model.input_size.height, model.input_size.width])
        tf_depth_resized = tf.image.resize_images(tf_depth, [model.output_size.height, model.output_size.width])

        # Create Tensors for Batch Training
        tf_batch_data_resized, tf_batch_data, tf_batch_labels = data.prepareTrainData(tf_image_resized,
                                                                                      tf_depth_resized, args.batch_size)

        # Build Network Model
        model.build_model(tf_batch_data, tf_batch_labels)
        model.build_losses()
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

    print("\n[Network/Training] Running built graph...")
    with tf.Session(graph=graph) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        # Check Dataset Integrity
        numSamples = data.checkIntegrity(sess, tf_train_image_filenames, tf_train_depth_filenames)

        # Proclaim the epochs
        epochs = np.floor(args.batch_size * args.max_steps / numSamples)
        print('\nTrain with approximately %d epochs' % epochs)

        # =================
        #  Training Loop
        # =================
        start = time.time()

        if args.show_train_progress:
            train_plotObj = Plot(args.mode, title='Train Predictions')

        if args.show_valid_progress:
            valid_plotObj = Plot(args.mode, title='Validation Prediction')

        # TODO: Adaptar
        # for i in range((len(dataloader.valid_dataset))):
        #     image, depth, image_crop, _ = dataloader.readImage(dataloader.valid_dataset[i],
        #                                               dataloader.valid_labels[i],
        #                                               mode='valid',
        #                                               showImages=False)

        #     valid_data_o[i] = image
        #     valid_labels_o[i] = depth
        #     valid_data_crop_o[i] = image_crop

        print("[Network/Training] Training Initialized!\n")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for step in range(args.max_steps):
            start2 = time.time()

            # ----- Session Run! ----- #
            # Training
            _, batch_data_resized, batch_data, batch_labels, batch_log_labels, batch_pred, model.train.loss, summary_str = sess.run(
                [model.train_step, tf_batch_data_resized, tf_batch_data, tf_batch_labels, model.train.tf_log_labels,
                 model.fcrn.get_output(), model.tf_loss, model.summary_op])

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
            valid_image = plt.imread(
                "/home/nicolas/Downloads/workspace/nicolas/data/residential_continuous/testing/imgs/residential_2011_09_26_drive_0019_sync_0000000384.png")
            valid_depth = plt.imread(
                "/home/nicolas/Downloads/workspace/nicolas/data/residential_continuous/testing/dispc/residential_2011_09_26_drive_0019_sync_0000000384.png")

            # FIXME: valid_loss = -1
            feed_dict_valid = {model.valid.tf_image: np.expand_dims(valid_image, axis=0),
                               model.valid.tf_depth: np.expand_dims(np.expand_dims(valid_depth, axis=0), axis=3)}
            valid_image, valid_pred, valid_labels, valid_log_labels = sess.run(
                [model.valid.tf_image_resized, model.fcrn_valid.get_output(), model.valid.tf_depth_resized,
                 model.valid.tf_log_depth_resized], feed_dict=feed_dict_valid)
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
                    train_plotObj.showTrainResults(raw=batch_data_resized[0],
                                                   label=batch_labels[0, :, :, 0],
                                                   log_label=batch_log_labels[0, :, :, 0],
                                                   pred=batch_pred[0, :, :, 0])

                if args.show_valid_progress:
                    valid_plotObj.showValidResults(raw=valid_image[0, :, :],
                                                   label=valid_labels[0, :, :, 0],
                                                   log_label=valid_log_labels[0, :, :, 0],
                                                   pred=valid_pred[0, :, :, 0])

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
    print('[%s] Selected Params: \n\n%s' % (appName, args))

    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    # -----------------------------------------
    #  Network Testing Model - Importing Graph
    # -----------------------------------------
    # Loads the dataset and restores a specified trained model.
    dataloader = Dataloader(args.data_path, args.dataset, args.mode)  # TODO: Usar leitura pelo Tensorflow

    # Create a placeholder for the input image
    tf_image = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = ResNet50UpProj({'data': tf_image}, batch_size, 1, False)

    # Memory Allocation
    # Length of test_dataset used, so when there is not test_labels, the variable will still be declared.
    pred = np.zeros((dataloader.numTestSamples, dataloader.outputSize[1], dataloader.outputSize[2]),
                    dtype=np.float32)  # (?, 43, 144)

    test_labels_o = np.zeros((len(dataloader.test_dataset), dataloader.outputSize[1], dataloader.outputSize[2]),
                             dtype=np.int32)  # (?, 43, 144)

    test_data_o = np.zeros(
        (len(dataloader.test_dataset), dataloader.inputSize[1], dataloader.inputSize[2], dataloader.inputSize[3]),
        dtype=np.uint8)  # (?, 172, 576, 3)

    test_data_crop_o = np.zeros(
        (len(dataloader.test_dataset), dataloader.inputSize[1], dataloader.inputSize[2], dataloader.inputSize[3]),
        dtype=np.uint8)  # (?, 172, 576, 3)

    with tf.Session() as sess:
        # Load the converted parameters
        print('\n[network/Testing] Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, args.model_path)

        # Use to load from npy file
        # net.load(model_data_path, sess)

        # ==============
        #  Testing Loop
        # ==============
        start = time.time()
        for i, image_path in enumerate(dataloader.test_dataset):
            start2 = time.time()

            if dataloader.test_labels:  # It's not empty
                image, depth, image_crop, depth_bilinear = dataloader.readImage(dataloader.test_dataset[i],
                                                                                dataloader.test_labels[i],
                                                                                mode='test')

                test_labels_o[i] = depth
                # test_labelsBilinear_o[i] = depth_bilinear # TODO: Usar?
            else:
                image, _, image_crop, _ = dataloader.readImage(dataloader.test_dataset[i], None, mode='test')

            test_data_o[i] = image
            test_data_crop_o[i] = image_crop

            # Evalute the network for the given image
            pred_temp = sess.run(net.get_output(),
                                 feed_dict={tf_image: np.expand_dims(np.asarray(test_data_o[i]), axis=0)})
            pred[i] = pred_temp[:, :, :, 0]

            # Prints Testing Progress
            end2 = time.time()
            print('step: %d/%d | t: %f' % (i + 1, dataloader.numTestSamples, end2 - start2))
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
        if dataloader.test_labels:
            metricsLib.evaluateTesting(pred, test_labels_o)
        else:
            print(
                "[Network/Testing] It's not possible to calculate Metrics. There are no corresponding labels for Testing Predictions!")

        # Show Results
        if args.show_test_results:
            test_plotObj = Plot(args.mode, title='Test Predictions')
            for i in range(dataloader.numTestSamples):
                test_plotObj.showTestResults(raw=test_data_crop_o[i],
                                             label=test_labels_o[i],
                                             log_label=np.log(test_labels_o[i] + LOSS_LOG_INITIAL_VALUE),
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
