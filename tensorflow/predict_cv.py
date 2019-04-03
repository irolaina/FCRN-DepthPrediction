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

from modules.framework import load_model
from modules.third_party.laina.fcrn import ResNet50UpProj
from modules.utils import settings

# ==================
#  Global Variables
# ==================
SAVE_IMAGES = False

global max_depth
global timer


# ===========
#  Functions
# ===========
def argument_handler():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
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


class maxDepth:
    def __init__(self):
        self.counter_len = 1000
        self.l_fps_history = [10 for _ in range(self.counter_len)]
        self.max_depth_counter = circular_counter(self.counter_len)

    def update(self, src):
        max_depth = np.max(src)
        self.l_fps_history[next(self.max_depth_counter) - 1] = max_depth
        return max_depth

    @property
    def get_avg(self):
        return sum(self.l_fps_history) / float(self.counter_len)


class CvTimer:
    def __init__(self):
        self.tick_frequency = cv2.getTickFrequency()
        self.tick_at_init = cv2.getTickCount()
        self.last_tick = self.tick_at_init
        self.counter_len = 100
        self.l_fps_history = [10 for _ in range(self.counter_len)]
        self.fps_counter = circular_counter(self.counter_len)

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
    def get_avg(self):
        return sum(self.l_fps_history) / float(self.counter_len)


def apply_overlay(frame, pred_jet_resized):
    alpha = 0.5
    background = frame.copy()
    overlay = pred_jet_resized.copy()
    overlay = cv2.addWeighted(background, alpha, overlay, 1 - alpha, 0)

    return overlay


def convertScaleAbs(src):
    global max_depth

    # print(max_depth.fps(src), max_depth.get_avg)
    max_depth.update(src)

    # return cv2.convertScaleAbs(src * (255 / np.max(src)))
    return cv2.convertScaleAbs(src * (255 / max_depth.get_avg))


def generate_colorbar(height, colormap, inv=False):
    colorbar = np.zeros(shape=(height, 45), dtype=np.uint8)

    for row in range(colorbar.shape[0]):
        for col in range(colorbar.shape[1]):
            # print(row, col)
            colorbar[row, col] = int((row / colorbar.shape[0]) * 255)

    if inv:
        colorbar = 255 - colorbar

    colorbar = cv2.applyColorMap(colorbar, colormap)

    cv2.putText(colorbar, "%0.2f-" % max_depth.get_avg, (1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255))
    cv2.putText(colorbar, "%0.2f-" % (max_depth.get_avg / 2), (1, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255))
    cv2.putText(colorbar, "%0.2f-" % 0.0, (10, height - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

    return colorbar


def process_images(frame, pred, remove_sky=False):
    global timer
    suffix = ''

    if remove_sky:
        # Remove Sky
        crop_height_perc = 0.3
        frame = frame[int(crop_height_perc * frame.shape[0]):, :, :]
        pred = pred[:, int(crop_height_perc * pred.shape[1]):, :, :]

        # print(frame.shape)
        # print(pred.shape)

        suffix = ' (without sky)'

    # Change Data Scale from meters to uint8
    pred_uint8 = cv2.convertScaleAbs(pred[0])
    pred_scaled_uint8 = convertScaleAbs(pred[0])

    # Apply Median Filter
    pred_median = cv2.medianBlur(pred[0], 3)
    pred_median_scaled_uint8 = convertScaleAbs(pred_median)

    # Change Colormap
    pred_jet = cv2.applyColorMap(255 - pred_median_scaled_uint8, cv2.COLORMAP_JET)
    pred_hsv = cv2.applyColorMap(pred_median_scaled_uint8, cv2.COLORMAP_HSV)

    # Resize
    pred_jet_resized = cv2.resize(pred_jet, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    pred_hsv_resized = cv2.resize(pred_hsv, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Apply the overlay
    overlay = apply_overlay(frame, pred_jet_resized)

    # Write text on Image
    cv2.putText(frame, "fps=%0.2f avg=%0.2f" % (timer.fps, timer.get_avg), (1, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255))

    # Generate Colorbar
    colorbar_jet = generate_colorbar(height=pred_jet_resized.shape[0], colormap=cv2.COLORMAP_JET)
    colorbar_hsv = generate_colorbar(height=pred_jet_resized.shape[0], colormap=cv2.COLORMAP_HSV, inv=True)

    # Concatenates Images
    conc = cv2.hconcat([pred_uint8, pred_scaled_uint8, pred_median_scaled_uint8])
    conc2 = cv2.hconcat([frame, pred_jet_resized, colorbar_jet, pred_hsv_resized, colorbar_hsv, overlay])

    # Debug
    if args.debug:
        print(pred.shape, pred.dtype)
        print(pred_uint8.shape, pred_uint8.dtype)
        print(pred_jet_resized.shape, pred_jet_resized.dtype)
        print(pred_hsv_resized.shape, pred_hsv_resized.dtype)
        print(overlay.shape, overlay.dtype)

    # Display the resulting frame - OpenCV
    # cv2.imshow('frame', frame)
    # cv2.imshow('pred', pred_uint8)
    # cv2.imshow('pred_jet (scaled, median, resized)', pred_jet_resized)
    # cv2.imshow('pred(scaled)', pred_scaled_uint8)
    # cv2.imshow('pred_hsv (scaled, median, resized)', pred_hsv_resized)
    # cv2.imshow('pred(scaled, median)', pred_median_scaled_uint8)
    # cv2.imshow('overlay', overlay)
    cv2.imshow('pred, pred(scaled), pred(scaled, median)' + suffix, conc)
    cv2.imshow('frame, pred_jet, pred_hsv, overlay' + suffix, conc2)


timer = CvTimer()
max_depth = maxDepth()
args = argument_handler()


# ======
#  Main
# ======
def main():
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
    tf_image_float32 = tf.image.convert_image_dtype(input_node, tf.float32)

    with tf.variable_scope('model'):  # Disable for running original models!!!
        # Construct the network
        net = ResNet50UpProj({'data': tf.expand_dims(tf_image_float32, axis=0)}, batch=batch_size, keep_prob=1.0,
                             is_training=False)

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

        # --------- #
        #  Restore  #
        # --------- #
        print('\n[network/Predicting] Loading the model...')
        load_model(saver=tf.train.Saver(), sess=sess)

        # Use to load from npy file
        # net.load(args.model_path, sess)

        count = 0
        while True:
            # Capture frame-by-frame
            _, frame = cap.read()
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            # Evaluate the network for the given image
            try:
                pred, pred_50, pred_80 = sess.run([tf_pred, tf_pred_50, tf_pred_80], feed_dict={input_node: frame})
                pred_50_uint8_scaled = convertScaleAbs(pred_50[0])
                pred_80_uint8_scaled = convertScaleAbs(pred_80[0])
                cv2.imshow('pred_50 (scaled)', pred_50_uint8_scaled)
                cv2.imshow('pred_80 (scaled)', pred_80_uint8_scaled)
            except UnboundLocalError:
                pred = sess.run(tf_pred, feed_dict={input_node: frame})

            # Debug
            if args.debug:
                print(frame)
                print(frame.shape, frame.dtype)
                print()
                print(pred)
                print(pred.shape, pred.dtype)
                input("Continue...")

            # ------------------ #
            #  Image Processing  #
            # ------------------ #
            # process_images(frame, pred) # FIXME: Para redes que não consideram o ceu, os valores da predição sujam o valor de max_depth
            process_images(frame, pred, remove_sky=True)

            # Save Images
            if SAVE_IMAGES:
                cv2.imwrite(settings.output_dir + "fcrn_cv/frame%06d.png" % count, frame)
                cv2.imwrite(settings.output_dir + "fcrn_cv/pred%06d.png" % count, pred_uint8)
                cv2.imwrite(settings.output_dir + "fcrn_cv/jet%06d.png" % count, pred_jet_resized)
                count += 1

            timer.reset()

            if cv2.waitKey(1) & 0xFF == ord('q'):  # without waitKey() the images are not shown.
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    print("Done.")
    sys.exit()


if __name__ == '__main__':
    main()
