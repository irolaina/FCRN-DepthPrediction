#!/usr/bin/python3
# -*- coding: utf-8 -*-

# ============
#  To-Do FCRN
# ============
# TODO: Terminar de Portar código de treinamento
# TODO: Terminar de Portar código de validacao
# TODO: Terminar de Portar código de testes

# TODO: Implementar leitura das imagens pelo Tensorflow - Treinamento
# TODO: Implementar leitura das imagens pelo Tensorflow - Validação
# TODO: Implementar leitura das imagens pelo Tensorflow - Treinamento

# TODO: Implementar Mask Out dos de valores válidos

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
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import glob
import time
import os
import warnings

import utils.args as argsLib
from utils.dataloader import Dataloader
import utils.loss as loss
from utils.plot import Plot
import models  # TODO: Change

# from PIL import Image
# from scipy import misc as scp
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


def predict(model_data_path, image_path):
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
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)

    with tf.Session() as sess:
        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        # net.load(model_data_path, sess)

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})

        # Plot result
        fig = plt.figure()
        ii = plt.imshow(pred[0, :, :, 0], interpolation='nearest')
        fig.colorbar(ii)
        plt.show()

        return pred


# ===================== #
#  Training/Validation  #
# ===================== #
def train(args, params):
    save_path, save_restore_path = createSaveFolder()

    graph = tf.Graph()
    with graph.as_default():
        # Default input size
        batch_size = 1

        # Create a placeholder for the input image
        dataloader = Dataloader(args.data_path, params, args.dataset, args.mode)
        params['inputSize'] = dataloader.inputSize
        params['outputSize'] = dataloader.outputSize

        # print(params['inputSize'],params['outputSize'])
        tf_image = tf.placeholder(tf.float32,
                                  shape=(None, params['inputSize'][1], params['inputSize'][2], params['inputSize'][3]))

        net = models.ResNet50UpProj({'data': tf_image}, params['batch_size'], 1, False)

        # Tensorflow Variables
        tf_labels = tf.placeholder(tf.float32,
                                   shape=(None, params['outputSize'][1], params['outputSize'][2]),
                                   name='labels')  # (?, 96, 288)

        tf_log_labels = tf.log(tf_labels + LOSS_LOG_INITIAL_VALUE,
                               name='log_labels')  # Just for displaying Image

        tf_global_step = tf.Variable(0, trainable=False,
                                     name='global_step')  # Count the number of steps taken.

        tf_learningRate = params['learning_rate']
        if params['ldecay']:
            tf_learningRate = tf.train.exponential_decay(tf_learningRate, tf_global_step, 1000, 0.95,
                                                         staircase=True, name='ldecay')

        loss_name, tf_loss = loss.tf_MSE(net.get_output(), tf_log_labels)

        optimizer = tf.train.AdamOptimizer(tf_learningRate)
        trainer = optimizer.minimize(tf_loss, global_step=tf_global_step)

        # Creates Saver Obj
        train_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()

        # =================
        #  Training Loop
        # =================
        start = time.time()

        if args.show_train_progress:
            train_plotObj = Plot(args.mode, title='Train Predictions')

        if args.show_valid_progress:
            valid_plotObj = Plot(args.mode, title='Validation Prediction')

        batch_data = np.zeros((args.batch_size,
                               params['inputSize'][1],
                               params['inputSize'][2],
                               params['inputSize'][3]),
                              dtype=np.float64)  # (?, 172, 576, 3)

        batch_data_crop = np.zeros((args.batch_size,
                                    params['inputSize'][1],
                                    params['inputSize'][2],
                                    params['inputSize'][3]),
                                   dtype=np.uint8)  # (?, 172, 576, 3)

        batch_labels = np.zeros((args.batch_size,
                                 params['outputSize'][1],
                                 params['outputSize'][2]),
                                dtype=np.int32)  # (?, 43, 144)

        for step in range(args.max_steps):
            start2 = time.time()

            # Training and Validation Batches and Feed Dictionary Preparation
            offset = (step * args.batch_size) % (dataloader.numTrainSamples - args.batch_size)  # Pointer
            batch_data_path = dataloader.train_dataset[offset:(offset + args.batch_size)]
            batch_labels_path = dataloader.train_labels[offset:(offset + args.batch_size)]

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

            # ----- Session Run! ----- #
            _, log_labels, train_pred, train_loss = sess.run([trainer, tf_log_labels, net.get_output(), tf_loss],
                                                             feed_dict=feed_dict_train)  # Training
            valid_loss = -1  # FIXME: value
            # -----

            # Prints Training Progress
            if step % 10 == 0:
                if args.show_train_progress:
                    train_plotObj.showTrainResults(raw=batch_data_crop[0, :, :], label=batch_labels[0, :, :],
                                                   log_label=log_labels[0, :, :],
                                                   pred=train_pred[0, :, :, 0])

                    # Plot.plotTrainingProgress(raw=batch_data_crop[0, :, :], label=batch_labels[0, :, :],log_label=log_labels[0, :, :], coarse=train_PredCoarse[0, :, :],fine=train_PredFine[0, :, :], fig_id=3)
                    pass

                if args.show_train_error_progress:
                    # FIXME:
                    # Plot.plotTrainingErrorProgress(raw=batch_data_crop[0, :, :], label=batch_labels[0, :, :],
                    #                                coarse=train_PredCoarse[0, :, :], fine=train_PredFine[0, :, :],
                    #                                figId=8)
                    pass

                if args.show_valid_progress:
                    # FIXME:
                    # valid_plotObj.showValidResults(raw=valid_data_crop_o[0, :, :, :], label=valid_labels_o[0],
                    #                                log_label=np.log(valid_labels_o[0] + LOSS_LOG_INITIAL_VALUE),
                    #                                coarse=valid_PredCoarse[0], fine=valid_PredFine[0])
                    pass

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

        # Pegar a função do bitboyslab que está mais completa
        # Logs the obtained test result
        f = open('results.txt', 'a')
        f.write("%s\t\t%s\t\t%s\t\t%s\t\tsteps: %d\ttrain_lossF: %f\tvalid_lossF: %f\t%f\n" % (
            datetime, args.model_name, args.dataset, loss_name, step, train_loss, valid_loss, sim_train))
        f.close()


# ========= #
#  Testing  #
# ========= #
def test(args, params):
    print('[%s] Selected mode: Test' % appName)
    print('[%s] Selected Params: %s' % (appName, args))

    # -----------------------------------------
    #  Network Testing Model - Importing Graph
    # -----------------------------------------
    # Loads the dataset and restores a specified trained model.
    dataloader = Dataloader(args.data_path, params, args.dataset, args.mode)
    model = ImportGraph(args.restore_path)

    # Memory Allocation
    # Length of test_dataset used, so when there is not test_labels, the variable will still be declared.
    predCoarse = np.zeros((dataloader.numTestSamples, dataloader.outputSize[1], dataloader.outputSize[2]),
                          dtype=np.float32)  # (?, 43, 144)

    predFine = np.zeros((dataloader.numTestSamples, dataloader.outputSize[1], dataloader.outputSize[2]),
                        dtype=np.float32)  # (?, 43, 144)

    test_labels_o = np.zeros((len(dataloader.test_dataset), dataloader.outputSize[1], dataloader.outputSize[2]),
                             dtype=np.int32)  # (?, 43, 144)

    test_data_o = np.zeros(
        (len(dataloader.test_dataset), dataloader.inputSize[1], dataloader.inputSize[2], dataloader.inputSize[3]),
        dtype=np.uint8)  # (?, 172, 576, 3)

    test_data_crop_o = np.zeros(
        (len(dataloader.test_dataset), dataloader.inputSize[1], dataloader.inputSize[2], dataloader.inputSize[3]),
        dtype=np.uint8)  # (?, 172, 576, 3)

    predCoarseBilinear = np.zeros((dataloader.numTestSamples, dataloader.inputSize[1], dataloader.inputSize[2]),
                                  dtype=np.float32)  # (?, 172, 576)

    predFineBilinear = np.zeros((dataloader.numTestSamples, dataloader.inputSize[1], dataloader.inputSize[2]),
                                dtype=np.float32)  # (?, 172, 576)

    # TODO: Usar?
    # test_labelsBilinear_o = np.zeros(
    #     (len(dataloader.test_dataset), dataloader.inputSize[1], dataloader.inputSize[2]),
    #     dtype=np.int32)  # (?, 172, 576)

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

        if APPLY_BILINEAR_OUTPUT:
            predCoarse[i], predFine[i], predCoarseBilinear[i], predFineBilinear[i] = model.networkPredict(image,
                                                                                                          APPLY_BILINEAR_OUTPUT)
        else:
            predCoarse[i], predFine[i] = model.networkPredict(image)

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
    if args.output_directory == '':
        output_directory = os.path.dirname(args.restore_path)
    else:
        output_directory = args.output_directory

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if SAVE_TEST_DISPARITIES:
        np.save(output_directory + 'test_coarse_disparities.npy', predCoarse)
        np.save(output_directory + 'test_fine_disparities.npy', predFine)

    # Calculate Metrics
    if dataloader.test_labels:
        metrics.evaluateTesting(predFine, test_labels_o)
    else:
        print(
            "[Network/Testing] It's not possible to calculate Metrics. There are no corresponding labels for Testing Predictions!")

    # Show Results
    if args.show_test_results:
        test_plotObj = Plot(args.mode, title='Test Predictions')
        for i in range(dataloader.numTestSamples):
            test_plotObj.showTestResults(test_data_crop_o[i], test_labels_o[i],
                                         np.log(test_labels_o[i] + LOSS_LOG_INITIAL_VALUE), predCoarse[i], predFine[i],
                                         i)


def main():
    # Parse arguments
    args = argsLib.argumentHandler()
    # args = argsLib.argumentHandler_original()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Predict the image
    # pred = predict(args.model_path, args.image_paths)

    modelParams = {'inputSize': -1,
                   'outputSize': -1,
                   'model_name': args.model_name,
                   'learning_rate': args.learning_rate,
                   'batch_size': args.batch_size,
                   'max_steps': args.max_steps,
                   'dropout': args.dropout,
                   'ldecay': args.ldecay,
                   'l2norm': args.l2norm,
                   'full_summary': args.full_summary}

    if args.mode == 'train':
        train(args, modelParams)
    elif args.mode == 'test':
        test(args, modelParams)

    print("\n[%s] Done." % appName)

    os._exit(0)


if __name__ == '__main__':
    main()
