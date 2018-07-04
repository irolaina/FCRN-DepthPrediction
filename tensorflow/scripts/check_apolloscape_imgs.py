import glob
import imageio
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf

data = np.genfromtxt("data/apolloscape_train.txt", dtype='str', delimiter='\t')
data = np.genfromtxt("data/apolloscape_test.txt", dtype='str', delimiter='\t')

root_path = "/media/nicolas/nicolas_seagate/datasets/apolloscape/data/"
images_filename = data[:, 0]
depths_filename = data[:, 1]

images_filename = [root_path + image_filename for image_filename in images_filename]
depths_filename = [root_path + depth_filename for depth_filename in depths_filename]

for i, image_filename in enumerate(images_filename):
    print(i, image_filename)
print(len(images_filename))

for j, depth_filename in enumerate(depths_filename):
    print(j, depth_filename)
print(len(depths_filename))

# images_filename.sort()
# depths_filename.sort()

numSamples = len(images_filename)


# Tensorflow Tensors
tf_image_key = tf.placeholder(tf.string)
tf_depth_key = tf.placeholder(tf.string)

tf_image_file = tf.read_file(tf_image_key)
tf_depth_file = tf.read_file(tf_depth_key)

tf_image_raw = tf.image.decode_jpeg(tf_image_file, channels=3)
tf_depth_raw = tf.image.decode_png(tf_depth_file, channels=1, dtype=tf.uint16)

with tf.Session() as sess:
    # ----- Tensors Initialization ----- #
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    invalidPairs = []
    for i in range(24450, numSamples): # TODO: Remover valor inicial para percorrer caso deseja-se percorrer todas as imagens
        print(i, images_filename[i], depths_filename[i])

        # Run Session
        feed_input = {tf_image_key: images_filename[i], tf_depth_key:depths_filename[i]}
        image, depth = sess.run([tf_image_raw, tf_depth_raw], feed_input)

        # Check if the pair is valid
        image_head, image_tail = os.path.split(images_filename[i])
        depth_head, depth_tail = os.path.split(depths_filename[i])

        if image_tail.split('_')[0] != depth_tail.split('_')[0]:
            invalidPairs.append([images_filename[i], depths_filename[i]])

        # print(image.shape)
        # print(depth.shape)

        plt.figure(1)
        plt.imshow(image)
        plt.figure(2)
        plt.imshow(depth[:, :, 0])
        plt.draw()
        plt.pause(2)
        # input()

print()
print("invalid_pairs:", invalidPairs)
print(len(invalidPairs))

print("Done")