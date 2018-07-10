#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Helpful Links
# http://kieleth.blogspot.com.br/2014/03/opencv-calculate-average-fps-in-python.html

# ===========
#  Libraries
# ===========
import argparse
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

from modules.third_party.laina.fcrn import ResNet50UpProj

# ==================
#  Global Variables
# ==================
SAVE_IMAGES = False


# ===========
#  Functions
# ===========
def argumentHandler():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help="Select which gpu to run the code", default='0')
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('video_path', help='Directory of images to predict')
    return parser.parse_args()


def circular_counter(max):
    """helper function that creates an eternal counter till a max value"""
    x = 0
    while True:
        if x == max:
            x = 0
        x += 1
        yield x


class CvTimer(object):
    def __init__(self):
        self.tick_frequency = cv2.getTickFrequency()
        self.tick_at_init = cv2.getTickCount()
        self.last_tick = self.tick_at_init
        self.fps_len = 100
        self.l_fps_history = [10 for x in range(self.fps_len)]
        self.fps_counter = circular_counter(self.fps_len)

    def reset(self):
        self.last_tick = cv2.getTickCount()

    @staticmethod
    def get_tick_now():
        return cv2.getTickCount()

    @property
    def fps(self):
        fps = self.tick_frequency / (self.get_tick_now() - self.last_tick)
        self.l_fps_history[next(self.fps_counter) - 1] = fps
        return fps

    @property
    def avg_fps(self):
        return sum(self.l_fps_history) / float(self.fps_len)


# ======
#  Main
# ======
def main():
    args = argumentHandler()

    timer = CvTimer()

    print(args.model_path)
    print(args.video_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Load from Camera or Video
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(args.video_path)

    if not cap.isOpened():  # Check if it succeeded
        print("It wasn't possible to open the camera.")
        return -1

    # ----------------
    #  Building Graph
    # ----------------
    # Default input size
    height, width, channels = 228, 304, 3
    batch_size = 1

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.uint8, shape=(height, width, channels))
    # tf_image_float32 = tf.cast(input_node, tf.float32)
    tf_image_float32 = tf.image.convert_image_dtype(input_node, tf.float32)

    with tf.variable_scope('model'): # Disable for running original models!!!
        # Construct the network
        net = ResNet50UpProj({'data': tf.expand_dims(tf_image_float32, axis=0)}, batch=batch_size, keep_prob=1, is_training=False)


    # ---------------
    #  Running Graph
    # ---------------
    with tf.Session() as sess:
        # Load the converted parameters
        print('\nLoading the model...')

        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, args.model_path)

        # Use to load from npy file
        # net.load(args.model_path, sess)

        isFirstTime = True
        success = True
        count = 0
        while True:
            # Capture frame-by-frame
            success, frame = cap.read()
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

            # Evalute the network for the given image
            pred = sess.run(net.get_output(), feed_dict={input_node: frame})

            # Debug
            # print(frame)
            # print(frame.shape, frame.dtype)
            # print()
            # print(pred)
            # print(pred.shape, pred.dtype)
            # input("Continue...")

            # Image Processing
            pred_uint8 = cv2.convertScaleAbs(pred[0])

            pred_jet = cv2.applyColorMap(pred_uint8, cv2.COLORMAP_JET)
            pred_hsv = cv2.applyColorMap(pred_uint8*5, cv2.COLORMAP_HSV)

            pred_jet_resized = cv2.resize(pred_jet, (304, 228), interpolation=cv2.INTER_CUBIC)
            cv2.putText(pred_jet_resized, "fps=%0.2f avg=%0.2f" % (timer.fps, timer.avg_fps), (1, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            pred_hsv_resized = cv2.resize(pred_hsv, (304, 228), interpolation=cv2.INTER_CUBIC)

            # print(pred)
            # print("min:", np.min(pred))
            # print("max:", np.max(pred))

            # print(pred_uint8[0,:,:,0])
            # print(np.min(pred_uint8))
            # print(np.max(pred_uint8))
            # print(pred_uint8.shape, pred_uint8.dtype)
            # input("pred_uint8")

            # print(pred_uint8.shape, pred_uint8.dtype)
            # print(pred_jet_resized.shape, pred_jet_resized.dtype)
            # print(pred_hsv_resized.shape, pred_hsv_resized.dtype)

            # Display the resulting frame - Matplotlib
            # plt.figure(1)
            # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # OpenCV uses BGR, Matplotlib uses RGB
            # plt.figure(2)
            # plt.imshow(pred[0, :, :, 0])
            # plt.pause(0.001)

            # Display the resulting frame - OpenCV
            cv2.imshow('frame', frame)
            cv2.imshow('pred', pred_uint8)
            cv2.imshow('pred_jet', pred_jet_resized)
            cv2.imshow('pred_hsv (scaled)', pred_hsv_resized)

            # Save Images
            if SAVE_IMAGES:
                cv2.imwrite("output/fcrn_cv/frame%06d.png" % count, frame)
                cv2.imwrite("output/fcrn_cv/pred%06d.png" % count, pred_uint8)
                cv2.imwrite("output/fcrn_cv/jet%06d.png" % count, pred_jet_resized)
                count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):  # without waitKey() the images are not shown.
                break

            timer.reset()

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    print("Done.")
    sys.exit()


if __name__ == '__main__':
    main()
