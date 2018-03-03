#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ============
#  To-Do FCRN
# ============

# TODO: Implementar leitura das imagens pelo Tensorflow - Treinamento
# TODO: Implementar leitura das imagens pelo Tensorflow - Validação
# TODO: Implementar leitura das imagens pelo Tensorflow - Treinamento

# TODO: Implementar Mask Out dos de valores válidos
# TODO: Implementar Bilinear

# ================
#  To-Do Monodeep
# ================
# TODO: Verificar se as tarefas abaixo ainda fazem sentido para o FCRN
# FIXME: Após uma conversa com o vitor, aparentemente tanto a saida do coarse/fine devem ser lineares, nao eh necessario apresentar o otimizar da Coarse e a rede deve prever log(depth), para isso devo converter os labels para log(y_)
# TODO: Validar Métricas.
# TODO: Adicionar mais topologias
# TODO: If detect Ctrl+C, save training state.
# TODO: Vitor sugeriu substituir a parte fully connected da rede por filtros deconvolucionais, deste modo pode-se fazer uma predicao recuperando o tamanho da imagem alem de ter muito menos parametros treinaveis na rede.


# ===========
#  Libraries
# ===========
import os
import sys
import time
import warnings
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc as scp

from collections import deque

import utils.args as argsLib
import utils.metrics as metricsLib

from utils.dataloader import Dataloader
import utils.loss as loss
from utils.plot import Plot
from utils.fcrn import ResNet50UpProj

from PIL import Image
# from skimage import exposure
# from skimage import dtype_limits
# from skimage import transform


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


# TODO: Move
def saveTrainedModel(save_path, session, saver, model_name):
    """ Saves trained model """
    # Creates saver obj which backups all the variables.
    print("[Network/Training] List of Saved Variables:")
    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        print(i)  # i.name if you want just a name

    file_path = saver.save(session, os.path.join(save_path, "model." + model_name))
    print("\n[Results] Model saved in file: %s" % file_path)

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
        print('Loading the model')

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
class EarlyStopping():
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
        # Load Dataset
        dataloader = Dataloader(args.data_path, args.dataset, args.mode) # TODO: Desativar assim que a leitura por tensorflow estiver funcionando

        # FCRN (Fully Convolutional Residual Network
        tf_image = tf.placeholder(tf.float32,
                                  shape=(None, dataloader.inputSize[1], dataloader.inputSize[2], dataloader.inputSize[3]))

        net = ResNet50UpProj({'data': tf_image}, args.batch_size, 1, False)

        tf_labels = tf.placeholder(tf.float32,
                                   shape=(None, dataloader.outputSize[1], dataloader.outputSize[2]),
                                   name='labels')  # (?, 96, 288)

        tf_log_labels = tf.log(tf_labels + LOSS_LOG_INITIAL_VALUE,
                               name='log_labels')  # Just for displaying Image

        tf_global_step = tf.Variable(0, trainable=False,
                                     name='global_step')  # Count the number of steps taken.

        tf_learningRate = args.learning_rate
        if args.ldecay:
            tf_learningRate = tf.train.exponential_decay(tf_learningRate, tf_global_step, 1000, 0.95,
                                                         staircase=True, name='ldecay')

        loss_name, tf_loss = loss.tf_MSE(net.get_output()[:,:,:,0], tf_log_labels)

        optimizer = tf.train.AdamOptimizer(tf_learningRate)
        trainer = optimizer.minimize(tf_loss, global_step=tf_global_step)

        # TODO: Enable Summaries
        # with tf.name_scope("Summaries"):
        #     # Summary Objects
        #     summary_writer = tf.summary.FileWriter(save_path + args.log_directory, graph)
        #     summary_op = tf.summary.merge_all('model_0')

        # Creates Saver Obj
        train_saver = tf.train.Saver()

    # ----------------------------------------
    #  Network Training Model - Running Graph
    # ----------------------------------------
    # Local Variables and Memory Allocation
    step = 0
    stop = EarlyStopping()

    train_loss, valid_loss = None, None

    batch_data = np.zeros((args.batch_size,
                           dataloader.inputSize[1],
                           dataloader.inputSize[2],
                           dataloader.inputSize[3]),
                          dtype=np.float64)  # (?, 172, 576, 3) # FIXME: Value

    batch_data_crop = np.zeros((args.batch_size,
                                dataloader.inputSize[1],
                                dataloader.inputSize[2],
                                dataloader.inputSize[3]),
                               dtype=np.uint8)  # (?, 172, 576, 3)  # FIXME: Value

    batch_labels = np.zeros((args.batch_size,
                             dataloader.outputSize[1],
                             dataloader.outputSize[2]),
                            dtype=np.int32)  # (?, 43, 144)  # FIXME: Value

    valid_data_o = np.zeros((len(dataloader.valid_dataset),
                             dataloader.inputSize[1],
                             dataloader.inputSize[2],
                             dataloader.inputSize[3]),
                            dtype=np.float64)  # (?, 172, 576, 3) # FIXME: Nao deveria ser uint8 para cada canal? # FIXME: Value

    valid_data_crop_o = np.zeros((len(dataloader.valid_dataset),
                                  dataloader.inputSize[1],
                                  dataloader.inputSize[2],
                                  dataloader.inputSize[3]),
                                 dtype=np.uint8)  # (?, 172, 576, 3) # FIXME: Value

    valid_labels_o = np.zeros((len(dataloader.valid_labels),
                               dataloader.outputSize[1],
                               dataloader.outputSize[2]),
                              dtype=np.int32)  # (?, 43, 144) # FIXME: Value

    print("\n[Network/Training] Running built graph...")
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()

        # Proclaim the epochs
        epochs = np.floor(args.batch_size * args.max_steps / dataloader.numTrainSamples)
        print('Train with approximately %d epochs' % epochs)

        # =================
        #  Training Loop
        # =================
        start = time.time()
        if args.show_train_progress:
            train_plotObj = Plot(args.mode, title='Train Predictions')

        if args.show_valid_progress:
            valid_plotObj = Plot(args.mode, title='Validation Prediction')

        for i in range((len(dataloader.valid_dataset))):
            image, depth, image_crop, _ = dataloader.readImage(dataloader.valid_dataset[i],
                                                      dataloader.valid_labels[i],
                                                      mode='valid',
                                                      showImages=False)

            valid_data_o[i] = image
            valid_labels_o[i] = depth
            valid_data_crop_o[i] = image_crop

        print("[Network/Training] Training Initialized!\n")
        for step in range(args.max_steps):
            start2 = time.time()

            # Training and Validation Batches and Feed Dictionary Preparation
            offset = (step * args.batch_size) % (dataloader.numTrainSamples - args.batch_size)  # Pointer
            batch_data_path = dataloader.train_dataset[offset:(offset + args.batch_size)]
            batch_labels_path = dataloader.train_labels[offset:(offset + args.batch_size)]

            # print("offset: %d/%d" % (offset,dataloader.numTrainSamples))
            # print(batch_data_path)
            # print(len(batch_data_path))

            for i in range(len(batch_data_path)):
                # FIXME: os tipos retornados das variaveis estao errados, quando originalmente eram uint8 e int32, lembrar que o placeholder no tensorflow é float32
                image, depth, image_crop, _ = dataloader.readImage(batch_data_path[i],
                                                                   batch_labels_path[i],
                                                                   mode='train',
                                                                   showImages=False)

                # print(image.dtype,depth.dtype, image_crop.dtype, depth_crop.dtype)

                batch_data[i] = image
                batch_labels[i] = depth
                batch_data_crop[i] = image_crop

            feed_dict_train = {tf_image: batch_data, tf_labels: batch_labels}

            feed_dict_valid = {tf_image: valid_data_o, tf_labels: valid_labels_o}

            # ----- Session Run! ----- #
            # Training
            _, batch_log_labels, batch_pred, train_loss = sess.run([trainer, tf_log_labels, net.get_output(), tf_loss],feed_dict=feed_dict_train)

            # Validation
            valid_log_labels, valid_pred, valid_loss = sess.run([tf_log_labels, net.get_output(), tf_loss], feed_dict=feed_dict_valid)
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
                    train_plotObj.showTrainResults(raw=batch_data_crop[0, :, :],
                                                   label=batch_labels[0, :, :],
                                                   log_label=batch_log_labels[0, :, :],
                                                   pred=batch_pred[0, :, :, 0])

                    # Plot.plotTrainingProgress(raw=batch_data_crop[0, :, :], label=batch_labels[0, :, :],log_label=log_labels[0, :, :], coarse=train_PredCoarse[0, :, :],fine=train_PredFine[0, :, :], fig_id=3)
                    pass

                if args.show_train_error_progress:
                    # FIXME:
                    # Plot.plotTrainingErrorProgress(raw=batch_data_crop[0, :, :], label=batch_labels[0, :, :],
                    #                                coarse=train_PredCoarse[0, :, :], fine=train_PredFine[0, :, :],
                    #                                figId=8)
                    pass

                if args.show_valid_progress:
                    valid_plotObj.showValidResults(raw=valid_data_crop_o[0, :, :, :],
                                                   label=valid_labels_o[0],
                                                   log_label=valid_log_labels[0, :, :],
                                                   pred=valid_pred[0,:,:,0])

                end2 = time.time()
                print('step: {0:d}/{1:d} | t: {2:f} | Batch trLoss: {3:>16.4f} | vLoss: {4:>16.4f} '.format(step,
                                                                                                            args.max_steps,
                                                                                                            end2 - start2,
                                                                                                            train_loss,
                                                                                                            valid_loss))

        end = time.time()
        sim_train = end - start
        print("\n[Network/Training] Training FINISHED! Time elapsed: %f s\n" % sim_train)

        # ==============
        #  Save Results
        # ==============
        if SAVE_TRAINED_MODEL:
            saveTrainedModel(save_restore_path, sess, train_saver, args.model_name)

        # Logs the obtained test result
        f = open('results.txt', 'a')
        f.write("%s\t\t%s\t\t%s\t\t%s\t\tsteps: %d\ttrain_loss: %f\tvalid_loss: %f\tt: %f s\n" % (
            datetime, args.model_name, args.dataset, loss_name, step, train_loss, valid_loss, sim_train))
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
        print('Loading the model')

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
            pred_temp = sess.run(net.get_output(), feed_dict={tf_image: np.expand_dims(np.asarray(test_data_o[i]), axis=0)})
            pred[i] = pred_temp[:,:,:,0]

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
            np.save(output_directory[:-7] + 'test_pred.npy', pred) # The indexing removes 'restore' from folder path

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
                                             log_label=np.log(test_labels_o[i] + LOSS_LOG_INITIAL_VALUE), # TODO: utilizar tf_log_label
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