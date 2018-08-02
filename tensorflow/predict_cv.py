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
from modules.utils import detect_available_models

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
    parser.add_argument('-r', '--model_path', help='Converted parameters for the model', default='')
    parser.add_argument('-i', '--video_path', help='Directory of images to predict')
    return parser.parse_args()


def circular_counter(max_value):
    """helper function that creates an eternal counter till a max value"""
    x = 0
    while True:
        if x == max_value:
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

    args.model_path = detect_available_models(args)

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

    with tf.variable_scope('model'):  # Disable for running original models!!!
        # Construct the network
        net = ResNet50UpProj({'data': tf.expand_dims(tf_image_float32, axis=0)}, batch=batch_size, keep_prob=1, is_training=False)

    tf_pred = net.get_output()

    if 'kitti' in args.model_path:
        tf_imask_50 = tf.where(tf_pred < 50.0, tf.ones_like(tf_pred), tf.zeros_like(tf_pred))
        tf_imask_80 = tf.where(tf_pred < 80.0, tf.ones_like(tf_pred), tf.zeros_like(tf_pred))

        tf_pred_50 = tf.multiply(tf_pred, tf_imask_50)
        tf_pred_80 = tf.multiply(tf_pred, tf_imask_80)

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

        count = 0
        while True:
            # Capture frame-by-frame
            _, frame = cap.read()
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            # Evalute the network for the given image
            try:
                pred, pred_50, pred_80 = sess.run([tf_pred, tf_pred_50, tf_pred_80], feed_dict={input_node: frame})
                pred_50_uint8_scaled = cv2.convertScaleAbs(pred_50[0] * (255 / np.max(pred[0])))
                pred_80_uint8_scaled = cv2.convertScaleAbs(pred_80[0] * (255 / np.max(pred[0])))
                cv2.imshow('pred_50 (scaled)', pred_50_uint8_scaled)
                cv2.imshow('pred_80 (scaled)', pred_80_uint8_scaled)
            except UnboundLocalError:
                pred = sess.run(tf_pred, feed_dict={input_node: frame})

            # Debug
            # print(frame)
            # print(frame.shape, frame.dtype)
            # print()
            # print(pred)
            # print(pred.shape, pred.dtype)
            # input("Continue...")

            # ------------------ #
            #  Image Processing  #
            # ------------------ #
            # Convert Predicted Depth to uint8 Image
            pred_uint8 = cv2.convertScaleAbs(pred[0])
            pred_scaled_uint8 = cv2.convertScaleAbs(pred[0] * (255 / np.max(pred[0])))

            # Apply Median Filter
            pred_median = cv2.medianBlur(pred[0], 3)
            pred_median_scaled_uint8 = cv2.convertScaleAbs(pred_median * (255 / np.max(pred_median)))
            pred_jet = cv2.applyColorMap(255 - pred_median_scaled_uint8, cv2.COLORMAP_JET)
            pred_hsv = cv2.applyColorMap(pred_median_scaled_uint8, cv2.COLORMAP_HSV)

            # Change Colormap
            pred_jet_resized = cv2.resize(pred_jet, (304, 228), interpolation=cv2.INTER_CUBIC)
            cv2.putText(pred_jet_resized, "fps=%0.2f avg=%0.2f" % (timer.fps, timer.avg_fps), (1, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            pred_hsv_resized = cv2.resize(pred_hsv, (304, 228), interpolation=cv2.INTER_CUBIC)

            # Apply the overlay
            alpha = 0.5
            background = frame.copy()
            overlay = pred_jet_resized.copy()

            overlay = cv2.addWeighted(background, alpha, overlay, 1 - alpha, 0)

            # Concatenates Images
            conc = cv2.hconcat([pred_uint8, pred_scaled_uint8, pred_median_scaled_uint8])
            conc2 = cv2.hconcat([frame, pred_jet_resized, pred_hsv_resized, overlay])

            # print(background.shape, background.dtype)
            # print(overlay.shape, overlay.dtype)
            # print(added_image.shape, added_image.dtype)

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
            # cv2.imshow('frame', frame)
            # cv2.imshow('pred', pred_uint8)
            # cv2.imshow('pred_jet (scaled, median, resized)', pred_jet_resized)
            # cv2.imshow('pred (scaled)', pred_scaled_uint8)
            # cv2.imshow('pred_hsv (scaled, median, resized)', pred_hsv_resized)
            # cv2.imshow('pred (scaled, median)', pred_median_scaled_uint8)
            # cv2.imshow('overlay', overlay)
            cv2.imshow('pred, pred(scaled), pred (scaled, median)', conc)
            cv2.imshow('frame, pred_jet, pred_hsv, overlay', conc2)

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
