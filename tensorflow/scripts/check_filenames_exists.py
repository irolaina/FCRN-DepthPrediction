import tensorflow as tf
import numpy as np
import os

data = np.genfromtxt('data/apolloscape_train.txt', dtype='str', delimiter='\t')
# data = np.genfromtxt('tmp/apolloscape_train_little.txt', dtype='str', delimiter='\t')
# data = np.genfromtxt('tmp/apolloscape_train_little2.txt', dtype='str', delimiter='\t')

image_filenames = list(data[:, 0])
depth_filenames = list(data[:, 1])

image_size = len(image_filenames)
depth_size = len(depth_filenames)

for i, image in enumerate(image_filenames):
    # print("%d/%d"  % (i, image_size))
    if not os.path.exists(image):
        print("Fail!")
        print(image)

for i, depth in enumerate(depth_filenames):
    # print("%d/%d" % (i, depth_size))
    if not os.path.exists(image):
        print("Fail!")
        print(depth)

print("Done.")