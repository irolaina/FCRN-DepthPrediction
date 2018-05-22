import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# FIXME: RandomShuffleError @ epoch: 1244

def get_var_name(**kwargs): return kwargs.keys()

# =================
#  Read Input File
# =================
# data = np.genfromtxt('data/apolloscape_train.txt', dtype='str', delimiter='\t')
# data = np.genfromtxt('tmp/apolloscape_train_little.txt', dtype='str', delimiter='\t')
data = np.genfromtxt('tmp/apolloscape_train_little2.txt', dtype='str', delimiter='\t')

image_filenames = list(data[:, 0])
depth_filenames = list(data[:, 1])

print(len(image_filenames))
print(len(depth_filenames))

# ============
#  Tensorflow
# ============
# Input/Batch Tensors
tf_train_image_filename_queue = tf.train.string_input_producer(image_filenames, shuffle=False) # TODO: capacity aqui deve ser igual a do batch ou maior?
tf_train_depth_filename_queue = tf.train.string_input_producer(depth_filenames, shuffle=False) # TODO: capacity aqui deve ser igual a do batch ou maior?

# Reads images
image_reader = tf.WholeFileReader()
tf_image_key, tf_image_file = image_reader.read(tf_train_image_filename_queue)
tf_depth_key, tf_depth_file = image_reader.read(tf_train_depth_filename_queue)

tf_image = tf.image.decode_jpeg(tf_image_file)
tf_depth = tf.image.decode_png(tf_depth_file, channels=1, dtype=tf.uint16)

# Retrieves Shape
tf_image.set_shape((2710, 3384, 3))
tf_depth.set_shape((2710, 3384, 1))

# ===============
#  Prepare Batch
# ===============
batch_size = 4
num_threads = 1
min_after_dequeue = 16
capacity = min_after_dequeue + num_threads * batch_size

# Select:
tf_batch_image, tf_batch_depth = tf.train.batch([tf_image, tf_depth], batch_size, num_threads, capacity)
# tf_batch_image, tf_batch_depth = tf.train.shuffle_batch([tf_image, tf_depth], batch_size, capacity, min_after_dequeue, num_threads)

# Print Tensors
print("\nTensors:")
print("tf_image_key: \t", tf_image_key)
print("tf_image_file: \t", tf_image_file)
print("tf_depth_key: \t", tf_depth_key)
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
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    for i in range(numEpochs):
        image_key, image, depth_key, depth = sess.run([tf_image_key, tf_image, tf_depth_key, tf_depth])
        print("image:")
        print(image_key, len(image_key))
        print(image.shape)
        print("depth:")
        print(depth_key, len(depth_key))
        print(depth.shape)

        # image, depth = sess.run([tf_image, tf_depth])
        #
        # plt.figure(1)
        # plt.imshow(image)
        # plt.draw()
        # plt.figure(2)
        # plt.imshow(depth[:, :, 0])
        # plt.draw()
        # plt.pause(1)
        # # input("enter")

        # batch_image, batch_depth = sess.run([tf_batch_image, tf_batch_depth])
        # plt.figure(1)
        # plt.imshow(batch_image[0])
        # plt.draw()
        # plt.figure(2)
        # plt.imshow(batch_depth[0, :, :, 0])
        # plt.draw()
        # plt.pause(0.001)
        # input("enter")

        print("%d/%d" % (i + 1, numEpochs))
        input("enter")
        print()

    coord.request_stop()
    coord.join(threads)

    print("Done.")
