#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ============
#  To-Do FCRN
# ============
# TODO: Implementar leitura das imagens pelo Tensorflow - Validação
# TODO: Implementar leitura das imagens pelo Tensorflow - Teste
# TODO: Validar Métricas.

# TODO: Implementar Bilinear
# TODO: Estou usando momento?
# TODO: If detect Ctrl+C, save training state.


# ===========
#  Libraries
# ===========
import os
import glob
import sys
import time
import warnings
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc as scp

from collections import deque
from PIL import Image

import utils.args as argsLib
import utils.metrics as metricsLib

from utils.dataloader import Dataloader
from utils.model import Model
from utils.fcrn import ResNet50UpProj
from utils.plot import Plot


# ==================
#  Global Variables
# ==================
# Choose the current Machine:
# machine = 'olorin'
machine = 'xps'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")  # Suppress Warnings

appName = 'fcrn'
datetime = time.strftime("%Y-%m-%d") + '_' + time.strftime("%H-%M-%S")

ENABLE_EARLY_STOP = True
SAVE_TRAINED_MODEL = True
ENABLE_TENSORBOARD = True
SAVE_TEST_DISPARITIES = True
APPLY_BILINEAR_OUTPUT = False

# Early Stop Configuration
AVG_SIZE = 20
MIN_EVALUATIONS = 1000
MAX_STEPS_AFTER_STABILIZATION = 10000
LOSS_LOG_INITIAL_VALUE = 0.1


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

    # Construct the network
    net = ResNet50UpProj({'data': tf_image}, batch_size, 1, False)

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


# TODO: move
class EarlyStopping:
    def __init__(self):
        # Local Variables
        self.movMeanLast = 0
        self.movMean = deque()
        self.stabCounter = 0

    def check(self, step, valid_loss):
        self.movMean.append(valid_loss)

        if step > AVG_SIZE:
            self.movMean.popleft()

        movMeanAvg = np.sum(self.movMean) / AVG_SIZE
        movMeanAvgLast = np.sum(self.movMeanLast) / AVG_SIZE

        if (movMeanAvg >= movMeanAvgLast) and step > MIN_EVALUATIONS:
            # print(step,stabCounter)

            self.stabCounter += 1
            if self.stabCounter > MAX_STEPS_AFTER_STABILIZATION:
                print("\nSTOP TRAINING! New samples may cause overfitting!!!")
                return 1
        else:
            self.stabCounter = 0

            self.movMeanLast = deque(self.movMean)


# ===================== #
#  Training/Validation  #
# ===================== #
def train(args):
    print('[%s] Selected mode: Train' % appName)
    print('[%s] Selected Params: \n\n%s' % (appName, args))

    # Local Variables
    save_path, save_restore_path = createSaveFolder()

    # -----------------------------------------
    #  Network Training Model - Building Graph
    # -----------------------------------------
    graph = tf.Graph()
    with graph.as_default():
        # ATTENTION! Since these tensors operate on a FifoQueue, using .eval() may misalign the pair (image, depth)!!!
        # Local Variables
        imageRawSize = [375, 1242]
        imageNetwork_InputSize = [228, 304]
        depthNetwork_OutputSize = [128, 160]

        seed = random.randint(0, 2 ** 31 - 1)

        # TODO: Ler outros Datasets
        # KittiRaw Residential Continuous
        # Image: (375, 1242, 3) uint8
        # Depth: (375, 1242)    uint8

        # Searches dataset images filenames and create queue objects
        if machine == 'olorin':
            tf_train_image_filename_list = tf.train.match_filenames_once(
                "../../mestrado_code/monodeep/data/residential_continuous/training/imgs/*.png")
            tf_train_depth_filename_list = tf.train.match_filenames_once(
                "../../mestrado_code/monodeep/data/residential_continuous/training/dispc/*.png")
        elif machine == 'xps':
            if args.dataset == 'kittiraw_residential_continuous':
                tf_train_image_filename_list = tf.train.match_filenames_once(
                    "../../data/residential_continuous/training/imgs/*.png")
                tf_train_depth_filename_list = tf.train.match_filenames_once(
                    "../../data/residential_continuous/training/dispc/*.png")

            elif args.dataset == 'nyudepth':
                image_filenames = []
                depth_filenames = []

                root_folder = "/media/nicolas/Nícolas/datasets/nyu-depth-v2/images/training/"

                # Finds input images and labels inside list of folders.
                for folder in glob.glob(root_folder+"*/"):
                    print(folder)
                    os.chdir(folder)

                    for file in glob.glob('*_colors.png'):
                        print(file)
                        image_filenames.append(folder+file)

                    for file in glob.glob('*_depth.png'):
                        print(file)
                        depth_filenames.append(folder+file)

                    print()

                print("Summary - Training Inputs")
                print("image_filenames: ", len(image_filenames))
                print("depth_filenames: ", len(depth_filenames))

                tf_train_image_filename_list = tf.placeholder(tf.string)
                tf_train_depth_filename_list = tf.placeholder(tf.string)

        # Creates Inputs Queue
        tf_train_image_filename_queue = tf.train.string_input_producer(tf_train_image_filename_list, shuffle=False, seed=seed)
        tf_train_depth_filename_queue = tf.train.string_input_producer(tf_train_depth_filename_list, shuffle=False, seed=seed)

        # Reads images
        image_reader = tf.WholeFileReader()
        tf_image_key, image_file = image_reader.read(tf_train_image_filename_queue)
        tf_depth_key, depth_file = image_reader.read(tf_train_depth_filename_queue)

        tf_image = tf.image.decode_image(image_file, channels=3)  # uint8
        tf_depth = tf.image.decode_image(depth_file, channels=1)  # uint8

        # FIXME: Kitti Original as imagens de disparidade são do tipo int32, no caso do kittiraw_residential_continous são uint8
        # Restores images structure (size, type)
        # Method 1
        tf_image.set_shape([imageRawSize[0], imageRawSize[1], 3])
        tf_depth.set_shape([imageRawSize[0], imageRawSize[1], 1])
        tf_image_resized = tf.cast(
            tf.image.resize_images(tf_image, [imageNetwork_InputSize[0], imageNetwork_InputSize[1]]), tf.uint8)
        tf_depth_resized = tf.cast(
            tf.image.resize_images(tf_depth, [depthNetwork_OutputSize[0], depthNetwork_OutputSize[1]]), tf.uint8)

        # Method 2
        # tf_image = tf.image.convert_image_dtype(tf_image, tf.float32)
        # tf_depth = tf.image.convert_image_dtype(tf_depth, tf.float32)
        # tf_image.set_shape([imageRawSize[0], imageRawSize[1], 3])
        # tf_depth.set_shape([imageRawSize[0], imageRawSize[1], 1])

        # Downsizes Input and Depth Images
        tf_image_resized = tf.image.resize_images(tf_image, [imageNetwork_InputSize[0], imageNetwork_InputSize[1]])
        tf_depth_resized = tf.image.resize_images(tf_depth, [depthNetwork_OutputSize[0], depthNetwork_OutputSize[1]])

        # ------------------- #
        #  Data Augmentation  #
        # ------------------- #
        # Copy
        tf_image_proc = tf_image_resized
        tf_depth_proc = tf_depth_resized

        def augment_image_pair(image, depth):
            # randomly flip images
            do_flip = tf.random_uniform([], 0, 1)
            image_aug = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(image), lambda: image)
            depth_aug = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(depth), lambda: depth)

            # randomly shift gamma
            random_gamma = tf.random_uniform([], 0.8, 1.2)
            image_aug = image_aug ** random_gamma

            # randomly shift brightness
            random_brightness = tf.random_uniform([], 0.5, 2.0)
            image_aug = image_aug * random_brightness

            # randomly shift color
            random_colors = tf.random_uniform([3], 0.8, 1.2)
            white = tf.ones([tf.shape(image)[0], tf.shape(image)[1]])
            color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
            image_aug *= color_image

            # saturate
            image_aug = tf.clip_by_value(image_aug, 0, 1)

            return image_aug, depth_aug

        # randomly augment images
        do_augment = tf.random_uniform([], 0, 1)
        tf_image_proc, tf_depth_proc = tf.cond(do_augment > 0.5,
                                                 lambda: augment_image_pair(tf_image_resized, tf_depth_resized),
                                                 lambda: (tf_image_resized, tf_depth_resized))

        # Normalizes Input
        tf_image_proc = tf.image.per_image_standardization(tf_image_proc)

        tf_image_resized_uint8 = tf.cast(tf_image_resized, tf.uint8)  # Visual purpose

        # Creates Training Batch Tensors
        tf_batch_data_resized, tf_batch_data, tf_batch_labels = tf.train.shuffle_batch(
            # [tf_image_key, tf_depth_key],           # Enable for Debugging the filename strings.
            [tf_image_resized_uint8, tf_image_proc, tf_depth_proc],  # Enable for debugging images
            batch_size=args.batch_size,
            num_threads=1,
            capacity=16,
            min_after_dequeue=0)

        model = Model(args, tf_batch_data, tf_batch_labels, 'train')

        # TODO: Mover para model.py
        # TODO: Enable Summaries
        # with tf.name_scope("Summaries"):
        #     # Summary Objects
        #     summary_writer = tf.summary.FileWriter(save_path + args.log_directory, graph)
        #     summary_op = tf.summary.merge_all('model_0')

        # TODO: Mover para model.py
        # Creates Saver Obj
        train_saver = tf.train.Saver()

    # ----------------------------------------
    #  Network Training Model - Running Graph
    # ----------------------------------------
    # Local Variables and Memory Allocation
    step = 0
    stop = EarlyStopping()

    train_loss, valid_loss = None, None

    print("\n[Network/Training] Running built graph...")
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        # TODO: Unificar session.run
        def checkDatasetIntegrity(tf_image_str_list, tf_depth_str_list):
            try:
                if args.dataset == 'kittiraw_residential_continuous':
                    feed_dict = None
                    image_replace = [b'/imgs/', b'']
                    depth_replace = [b'/dispc/', b'']

                elif args.dataset == 'nyudepth':
                    feed_dict = {tf_image_str_list: image_filenames,
                                 tf_depth_str_list: depth_filenames}
                    image_replace = ['_colors.png', '']
                    depth_replace = ['_depth.png', '']

                image_str_list, depth_str_list = sess.run(
                    [tf_image_str_list, tf_depth_str_list], feed_dict=feed_dict)

                image_str_list_aux = [item.replace(image_replace[0], image_replace[1]) for item in
                                             image_str_list]
                depth_str_list_aux = [item.replace(depth_replace[0], depth_replace[1]) for item in
                                             depth_str_list]

                # print(image_str_list)
                # input("oi3")
                # print(depth_str_list)
                # input("oi4")
                #
                # print(image_str_list_aux)
                # input("oi3")
                # print(depth_str_list_aux)
                # input("oi4")

                numSamples = len(image_str_list_aux)

                print("[monodeep/Dataset] Checking if RGB and Depth images are paired... ")
                if image_str_list_aux == depth_str_list_aux:
                    print("[monodeep/Dataset] Check Integrity: Pass")
                    # del image_str_list, depth_str_list
                    # del image_str_list_aux, depth_str_list_aux
                else:
                    raise ValueError

                return numSamples, feed_dict

            except ValueError:
                print("[monodeep/Dataset] Check Integrity: Failed")
                raise SystemExit

        # Check Dataset Integrity
        numSamples, feed_dict_strings = checkDatasetIntegrity(tf_train_image_filename_list, tf_train_depth_filename_list)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

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
        for step in range(args.max_steps):
            start2 = time.time()

            # ----- Session Run! ----- #
            # Training
            if args.dataset == 'kittiraw_residential_continuous':
                _, batch_data_resized, batch_data, batch_labels, batch_log_labels, batch_pred, train_loss = sess.run(
                [model.train, tf_batch_data_resized, tf_batch_data, tf_batch_labels, model.tf_log_labels, model.fcrn.get_output(), model.tf_loss])

            elif args.dataset == 'nyudepth':
                image_key = sess.run([tf_image_key], feed_dict=feed_dict_strings)

                input("oi5")

                _, batch_data_resized, batch_data, batch_labels, batch_log_labels, batch_pred, train_loss = sess.run(
                [model.train, tf_batch_data_resized, tf_batch_data, tf_batch_labels, model.tf_log_labels, model.fcrn.get_output(), model.tf_loss])

            # _, batch_data_resized, batch_data, batch_labels, batch_log_labels, batch_pred, train_loss, images_resized, depths_resized, images_proc, depths_proc = sess.run(
            #     [train, tf_batch_data_resized, tf_batch_data, tf_batch_labels, tf_log_labels, net.get_output(), tf_loss, tf_image_resized, tf_depth_resized, tf_image_proc,
            #      tf_depth_proc])

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
            valid_loss = -1
            # valid_log_labels, valid_pred, valid_loss = sess.run([tf_log_labels, net.get_output(), tf_loss])
            # input("valid")
            # -----

            # TODO: Reativar código abaixo
            # if ENABLE_TENSORBOARD:
            #     # Write information to TensorBoard
            #     summary_writer.add_summary(summary_str, step)
            #     summary_writer.flush()  # Don't forget this command! It makes sure Python writes the summaries to the log-file

            # TODO: Validar
            if ENABLE_EARLY_STOP:
                if stop.check(step, valid_loss):
                    break

            # Prints Training Progress
            if step % 10 == 0:
                if args.show_train_progress:
                    train_plotObj.showTrainResults(raw=batch_data_resized[0],
                                                   label=batch_labels[0, :, :, 0],
                                                   log_label=batch_log_labels[0, :, :, 0],
                                                   pred=batch_pred[0, :, :, 0])

                if args.show_valid_progress:
                    valid_plotObj.showValidResults(raw=valid_data_crop_o[0, :, :, :],
                                                   label=valid_labels_o[0],
                                                   log_label=valid_log_labels[0, :, :],
                                                   pred=valid_pred[0, :, :, 0])

                end2 = time.time()
                print('step: {0:d}/{1:d} | t: {2:f} | Batch trLoss: {3:>16.4f} | vLoss: {4:>16.4f} '.format(step,
                                                                                                            args.max_steps,
                                                                                                            end2 - start2,
                                                                                                            train_loss,
                                                                                                            valid_loss))

        coord.request_stop()
        coord.join(threads)

        end = time.time()
        sim_train = end - start
        print("\n[Network/Training] Training FINISHED! Time elapsed: %f s\n" % sim_train)

        # ==============
        #  Save Results
        # ==============
        if SAVE_TRAINED_MODEL:
            model.saveTrainedModel(save_restore_path, sess, train_saver, args.model_name)

        # Logs the obtained test result
        f = open('results.txt', 'a')
        f.write("%s\t\t%s\t\t%s\t\t%s\t\tsteps: %d\ttrain_loss: %f\tvalid_loss: %f\tt: %f s\n" % (
            datetime, args.model_name, args.dataset, model.loss_name, step, train_loss, valid_loss, sim_train))
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
    dataloader = Dataloader(args.data_path, args.dataset, args.mode)

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
