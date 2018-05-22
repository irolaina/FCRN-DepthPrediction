# ========
#  README
# ========
# FIXME: RandomShuffleError
# Informações
# - Acontece na epoch: 1244, batch_size = 4
# - Acontece na epoch: 622, batch_size = 8
# - Não acontece com batch de strings
# - Não acontece com as images uint8, apenas com as depth uint16

# Testes
# TODO: [Ok] Batchs de Strings usando train.batch
# TODO: [Ok] Batchs de Strings usando train.random_shuffle_batch

# TODO: [  ] Batchs de Imagens usando train.batch
# TODO: [  ] Batchs de Imagens usando train.random_shuffle_batch

# TODO: [  ] Batchs de Imagens usando train.batch em funções separadas para image e depth -> DESSINCRONIZA!
# TODO: [  ] Batchs de Imagens usando train.random_shuffle_batch em funções separadas para image e depth -> DESSINCRONIZA!

# Dúvidas
# TODO: Por que string_input_producer sempre começa do segundo sample?
# TODO: Será que alguma imagem da lista de imagens não existe? Usei o check_filenames_exists.py e todas as imagens existem
# TODO: Será que o erro está sendo causado por tf_image ser uint8 e tf_depth ser uint16? Mudei pra float32 e continuou dando erro
# TODO: Um dos erros que deu indicava que talvez uma das imagens de depth tivesse formato inválido, isto é, que não fosse PNG.

# ===========
#  Libraries
# ===========
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==============
#  Batch Config
# ==============
batch_size = 8
num_threads = 4
min_after_dequeue = 16
capacity = min_after_dequeue + num_threads * batch_size

# =================
#  Read Input File
# =================
# Select Dataset:
# dataset_name = 'apolloscape'
dataset_name = 'kitticontinuous_residential'

if dataset_name == 'apolloscape':
    data = np.genfromtxt('data/apolloscape_train.txt', dtype='str', delimiter='\t')
    # data = np.genfromtxt('tmp/apolloscape_train_little.txt', dtype='str', delimiter='\t')
    # data = np.genfromtxt('tmp/apolloscape_train_little2.txt', dtype='str', delimiter='\t')

    image_shape = (2710, 3384, 3)
    depth_shape = (2710, 3384, 1)
elif dataset_name == 'kitticontinuous_residential':
    data = np.genfromtxt('data/kitticontinuous_residential_train.txt', dtype='str', delimiter='\t')

    image_shape = (375,1242,3)
    depth_shape = (375,1242,1)
else:
    raise SystemExit

image_filenames = list(data[:, 0])
depth_filenames = list(data[:, 1])

# print(image_filenames)
print(len(image_filenames))
# input("image_filenames")
# print(depth_filenames)
print(len(depth_filenames))
# input("depth_filenames")

# ============
#  Tensorflow
# ============
# Strings input tensors
tf_train_image_filename_queue = tf.train.string_input_producer(image_filenames, shuffle=False, capacity=capacity) # TODO: capacity aqui deve ser igual a do batch ou maior?
tf_train_depth_filename_queue = tf.train.string_input_producer(depth_filenames, shuffle=False, capacity=capacity) # TODO: capacity aqui deve ser igual a do batch ou maior?

# Reads/Decodes images
image_reader = tf.WholeFileReader()
tf_image_key, tf_image_file = image_reader.read(tf_train_image_filename_queue)
tf_depth_key, tf_depth_file = image_reader.read(tf_train_depth_filename_queue)

if dataset_name == 'apolloscape':
    tf_image = tf.image.decode_jpeg(tf_image_file)
    tf_depth = tf.image.decode_png(tf_depth_file, channels=1, dtype=tf.uint16)
elif dataset_name == 'kitticontinuous_residential':
    tf_image = tf.image.decode_png(tf_image_file, channels=3, dtype=tf.uint8)
    tf_depth = tf.image.decode_png(tf_depth_file, channels=1, dtype=tf.uint8)
else:
    raise SystemExit

# Retrieves Shape
tf_image.set_shape(image_shape)
tf_depth.set_shape(depth_shape)

# Change Type
# tf_image = tf.cast(tf_image, tf.float32) # uint8  -> float32
# tf_depth = tf.cast(tf_depth, tf.float32) # uint16 -> float32

# tf_image_shape = tf.shape(tf_image) # Test 6
# tf_depth_shape = tf.shape(tf_depth)

# =========================
#  Prepare Batch - Select:
# =========================
# Strings Batch
# tf_batch_image, tf_batch_depth = tf.train.batch([tf_image_key, tf_depth_key], batch_size, num_threads, capacity)  # Test 2
# tf_batch_image, tf_batch_depth = tf.train.shuffle_batch([tf_image_key, tf_depth_key], batch_size, capacity, min_after_dequeue, num_threads)  # Test 3

# Images Batch
# Junto
tf_batch_image, tf_batch_depth = tf.train.batch([tf_image, tf_depth], batch_size, num_threads, capacity, shapes=[image_shape, depth_shape]) # Test 4
# tf_batch_image, tf_batch_depth = tf.train.shuffle_batch([tf_image, tf_depth], batch_size, capacity, min_after_dequeue, num_threads) # Test 5

# Separado
# tf_batch_image = tf.train.batch([tf_image], batch_size, num_threads, capacity) # Test 4
# tf_batch_depth = tf.train.batch([tf_depth], batch_size, num_threads, capacity) # Test 4
# tf_batch_image = tf.train.shuffle_batch([tf_image], batch_size, capacity, min_after_dequeue, num_threads) # Test 5
# tf_batch_depth = tf.train.shuffle_batch([tf_depth], batch_size, capacity, min_after_dequeue, num_threads) # Test 5

# Print Tensors
print("\nTensors:")
print("tf_image_key: \t", tf_image_key)
print("tf_depth_key: \t", tf_depth_key)
print("tf_image_file: \t", tf_image_file)
print("tf_depth_file: \t", tf_depth_file)
print("tf_image: \t", tf_image)
print("tf_depth: \t", tf_depth)
print("tf_batch_image: ", tf_batch_image)
print("tf_batch_depth: ", tf_batch_depth)
input("enter")

# =====
#  Run
# =====
numEpochs = 2000
# numEpochs = 102687 # numSamples
with tf.Session() as sess:
    # ----- Tensors Initialization ----- #
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # ----- Main Loop ----- #
    i = 0
    while not coord.should_stop() and i < numEpochs:
        image_key, image, depth_key, depth = sess.run([tf_image_key, tf_image, tf_depth_key, tf_depth])
        print("image:")
        print(image_key, len(image_key))
        print(image.shape)
        print("depth:")
        print(depth_key, len(depth_key))
        print(depth.shape)

        # Test 2 e 3
        # batch_image, batch_depth = sess.run([tf_batch_image, tf_batch_depth])
        # batch_image = sess.run([tf_batch_image])
        # batch_depth = sess.run([tf_batch_depth])

        # print("batch_image:")
        # print(batch_image)
        # print("batch_depth:")
        # print(batch_depth)

        # image, depth = sess.run([tf_image, tf_depth])
        # plt.figure(1)
        # plt.imshow(image)
        # plt.draw()
        # plt.figure(2)
        # plt.imshow(depth[:, :, 0])
        # plt.draw()
        # plt.pause(1)

        # Test 4 e 5
        # batch_image, batch_depth = sess.run([tf_batch_image, tf_batch_depth])
        # plt.figure(1)
        # plt.imshow(batch_image[0])
        # plt.draw()
        # plt.figure(2)
        # plt.imshow(batch_depth[0, :, :, 0])
        # plt.draw()
        # plt.pause(0.001)

        # Test 6
        # image_shape, depth_shape = sess.run([tf_image_shape, tf_depth_shape])
        # print(image_shape)
        # print(depth_shape)

        i += 1
        print("%d/%d" % (i, numEpochs))
        # input("enter")
        print()

    # ----- Finish ----- #
    coord.request_stop()

    coord.join(threads)
    sess.close()
print("Done.")