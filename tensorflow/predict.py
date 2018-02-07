import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from glob import glob
import scipy.io as scio

import models

def predict(model_data_path, image_path, output_folder, show_plot=True, output_format="npy"):


    # Setup input and output
    images = sorted(glob(image_path))
    if output_folder and not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    o_formats = output_format.split(',')

    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

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
        #net.load(model_data_path, sess)

        for i, img_path in enumerate(images):
            print("Processing image {} / {}".format(i+1,len(images)))

            # Read image
            img = Image.open(img_path)
            img = img.resize([width,height], Image.ANTIALIAS)
            img = np.array(img).astype('float32')
            img = np.expand_dims(np.asarray(img), axis = 0)

            # Evalute the network for the given image
            pred = sess.run(net.get_output(), feed_dict={input_node: img})
            pred = np.squeeze(pred)

            # Write result
            if output_folder:
                input_filename, ext = os.path.splitext(os.path.basename(img_path))
                filename = os.path.join(output_folder, "{}_depth".format(input_filename))

                if "mat" in o_formats:
                    scio.savemat(filename,
                               {'depth': pred},
                               do_compression=True)

                if "npy" in o_formats:
                    np.save(filename, pred)

                if "img" in o_formats:
                    fig = plt.figure()
                    ii = plt.imshow(pred, interpolation='nearest')
                    fig.colorbar(ii)
                    plt.savefig(filename)
                    plt.close(fig)

            # Plot result
            if show_plot:
                fig = plt.figure()
                ii = plt.imshow(pred, interpolation='nearest')
                fig.colorbar(ii)
                plt.show()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_path', help='Images to predict, can be glob pattern')
    parser.add_argument('-o', '--output_folder', type=str, default=None,
                        help='Path to output depth maps')
    parser.add_argument('-f', '--output_format', type=str, default="npy",
                        help='output format as comma separated list, can be img, npy or mat')
    parser.add_argument('-p', '--plot', type=bool, default=False,
                        help="Plot output image with colorbar on the screen")
    args = parser.parse_args()

    # Predict the image
    predict(args.model_path, args.image_path, args.output_folder, args.plot, args.output_format)

    os._exit(0)

if __name__ == '__main__':
    main()
