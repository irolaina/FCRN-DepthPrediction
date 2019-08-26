import argparse
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

from modules.third_party.laina.fcrn import ResNet50UpProj


def predict(model_data_path, image_path):
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    # Read image
    img_raw = Image.open(image_path)
    img = img_raw.resize([width, height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis=0)

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = ResNet50UpProj({'data': input_node}, batch_size, 1, False)

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
        plt.figure()
        plt.imshow(img_raw)

        fig = plt.figure()
        ii = plt.imshow(pred[0, :, :, 0], interpolation='nearest')
        fig.colorbar(ii)

        plt.show()

        return pred


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help="Select which gpu to run the code", default='0')
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Predict the image
    pred = predict(args.model_path, args.image_paths)

    # noinspection PyProtectedMember
    os._exit(1)


if __name__ == '__main__':
    main()