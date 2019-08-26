from __future__ import absolute_import, division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # Do other imports now...

#IMPORT
import skimage.io as io
import cv2
import numpy as np
import glob
import sys, os
import tensorflow as tf
import h5py
from tensorflow import keras
import imageio
import matplotlib.pyplot as plt
import os
import time
import math
import warnings
import scipy
import argparse
import matplotlib as mpl
import argparse
from PIL import Image
import cv2
from scipy import interpolate
from scipy.interpolate import griddata
from math import (floor, ceil)
from tqdm import tqdm

def lin_interp(shape, xyd):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = scipy.interpolate.LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity

def load_txt():
    if not (os.path.exists('data/kitti_depth_train.txt') and os.path.exists('data/kitti_depth_val.txt')):  # and os.path.exists('kitti_completion_test(2).txt')):

        timer1 = -time.time()

        
        timer1 += time.time()

    else:

        timer1 = -time.time()

        try:

            def read_text_file(filename,dataset_path):
                print("\n[Dataloader] Loading '%s'..." % filename)
                try:
                    data = np.genfromtxt(filename,dtype='str',delimiter='\t')
                    # print(data.shape)

                    # Parsing Data
                    image_filenames = list(data[:,0])
                    depth_filenames = list(data[:,1])

                    timer = -time.time()
                    image_filenames = [dataset_path + filename for filename in image_filenames]
                    depth_filenames = [dataset_path + filename for filename in depth_filenames]
                    timer += time.time()
                    print('time:',timer,'s\n')

                except OSError:
                    raise OSError("Could not find the '%s' file." % filename)

                return image_filenames,depth_filenames

            image_filenames,depth_filenames = read_text_file(
                filename='/home/nicolas/MEGA/workspace/FCRN-DepthPrediction/tensorflow/data/kitti_depth_train.txt',
                dataset_path='/media/nicolas/nicolas_seagate/datasets/kitti/')

            image = sorted(image_filenames)
            depth = sorted(depth_filenames)

            train_images = image
            train_labels = depth

            print(len(image))
            print(len(depth))

            timer1 += time.time()

        except OSError:
            raise SystemExit

    return train_images, train_labels

def load_depth_map(filepath, k=2):
    image_input = Image.open(filepath)
    image_input = np.array(image_input)
    image_input = image_input.astype(np.float32)

    kernel = np.ones((k,k),np.uint8)
    closing = cv2.morphologyEx(image_input,cv2.MORPH_CLOSE,kernel)

    # Fix Data shift caused by close operation
    if k % 2 == 0:
        rows, cols = closing.shape
        tx,ty = -1,-1  # Offsets
        M = np.float32([[1,0,tx],[0,1,ty]])
        closing = cv2.warpAffine(closing,M,(cols,rows))
    closing = closing.astype(np.uint16)
    return image_input, closing

# ==============================
# Interpolation
# ==============================
def inter(depth):
    fig,axs = plt.subplots(4,1)
    for ax,interp in zip(axs,['None','nearest','bilinear','bicubic']):
        ax.imshow(depth,interpolation=interp)
        ax.set_title(interp.capitalize())
        ax.axis('off')
    plt.show()

if __name__ == '__main__':

    train_images, train_labels = load_txt()

    num_filenames = len(train_labels)

    k = 16
    depth_map, closing = load_depth_map(train_labels[0], k)

    # Number of Valid Pixels
    print(depth_map[depth_map > 0.0].size, closing[closing > 0.0].size)

    def showImage(img, title):
        plt.figure()
        plt.imshow(img)
        plt.title(title)
        plt.colorbar()

    showImage(closing, "close(k={})".format(k))
    showImage(depth_map, "Input")
    showImage(closing-depth_map, "Artefacts")
    plt.show()

    # cv2.imwrite('/media/usb/morph',closing)

    # for i in tqdm(train_labels):
    #
    #     depth_map, closing = load_depth_map(i)
    #
    #     print(depth_map.shape)
    #
    #     input("ok")
    #
    #     disp = lin_interp(depth_map.shape,depth_map)
    #
    #     print(disp.shape)
    #
    #     input("ok")
    #
    #     d = os.path.dirname(i)[40:]
    #
    #     best_file_name = '/media/usb'
    #
    #     save_root = os.path.join(os.path.dirname(best_file_name),'usb')
    #
    #     path_aux = os.path.join(save_root,d)
    #
    #     if not os.path.exists(path_aux):
    #         os.makedirs(path_aux)
    #
    #     path = os.path.join(path_aux,os.path.basename(i))
    #
    #     cv2.imwrite(path,closing)

    print("Done.")





