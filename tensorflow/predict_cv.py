# ===========
#  Libraries
# ===========
import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
import subprocess

from matplotlib import pyplot as plt
from PIL import Image

import models

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

# ======
#  Main
# ======
def main():
    args = argumentHandler()

    print(args.model_path)
    print(args.video_path)
    # input("args")

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Load from Camera or Video
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(args.video_path)

    # ----------------
    #  Building Graph
    # ----------------
    # Default input size
    height, width, channels = 228, 304, 3
    batch_size = 1

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    
    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
    tf_pred = tf.exp(net.get_output(), 'pred')

    # ---------------
    #  Running Graph
    # ---------------
    # files = []
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
        while(success):
        # for i in range(20):  # 20 frames
            # Capture frame-by-frame
            success, frame = cap.read()
            frame = cv2.resize(frame,(304, 228), interpolation = cv2.INTER_CUBIC)

            # Read image
            img = np.array(frame).astype('float32')
            img = np.expand_dims(np.asarray(img), axis=0)

            # Evalute the network for the given image
            pred_log, pred = sess.run([net.get_output(), tf_pred], feed_dict={input_node: img})

            # print(frame.shape, frame.dtype)
            # print()
            # print(pred_log)
            # print(type(pred_log))
            # print(pred_log.shape, pred_log.dtype)
            # input("enter")
            # print()
            # print(pred)      
            # print(type(pred))
            # print(pred.shape, pred.dtype)
            # input("enter2")

            # Barely Observed, Pred range (0, 12000)
            def check_min_max_values():
                min = np.min(pred)
                max = np.max(pred)

                if isFirstTime:
                    min_min = min 
                    max_max = max
                    isFirstTime = False

                if min < min_min:
                    min_min = min
                
                if max > max_max:
                    max_max = max

                print()
                print("min:", min)      
                print("max:", max)
                
            # check_min_max_values()
            
            # pred_uint8 = pred*(255.0/12000.0)
            # pred_uint8 = pred_uint8.astype(np.uint8)
            pred_uint8 = cv2.convertScaleAbs(pred)

            # print()
            # print(pred_uint8[0,:,:,0])
            # print(np.min(pred_uint8))
            # print(np.max(pred_uint8))
            # print(pred_uint8.shape, pred_uint8.dtype)
            # input("pred_uint8")

            # Display the resulting frame - Matplotlib
            # fname = '_tmp%03d.png' % i
            # print('Saving frame', fname)

            # plt.figure(1)
            # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # OpenCV uses BGR, Matplotlib uses RGB
            # plt.savefig(fname)
            # files.append(fname)
            # plt.figure(2)
            # plt.imshow(pred[0, :, :, 0])
            # plt.pause(0.001)

            # Display the resulting frame - OpenCV
            cv2.imshow('frame',frame)
            cv2.imshow('pred', pred_uint8[0, :, :, 0]) # FIXME: white screen

            if cv2.waitKey(1) & 0xFF == ord('q'): # without waitKey() the images are not shown.
                break

    # Encoding Video
    # print('Making movie animation.mpg - this may take a while')
    # subprocess.call("mencoder 'mf://_tmp*.png' -mf type=png:fps=10 -ovc lavc "
    #                 "-lavcopts vcodec=wmv2 -oac copy -o animation.mpg", shell=True)

    # CleanUp
    # for fname in files:
    #     os.remove(fname)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    os._exit(0)

    print("Done.")

if __name__ == '__main__':
    main()
