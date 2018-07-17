import imageio
import matplotlib.pyplot as plt
import os
import numpy as np

data = np.genfromtxt("../data/nyudepth_train.txt", dtype='str', delimiter='\t')
data = np.genfromtxt("../data/nyudepth_test.txt", dtype='str', delimiter='\t')

root_path = "/media/nicolas/nicolas_seagate/datasets/nyu-depth-v2/data/images/"
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

invalidPairs = []
for i in range(numSamples):
    print(i, images_filename[i], depths_filename[i])
    image_head, image_tail = os.path.split(images_filename[i])
    depth_head, depth_tail = os.path.split(depths_filename[i])

    if image_tail.split('_')[0] != depth_tail.split('_')[0]:
        invalidPairs.append([images_filename[i], depths_filename[i]])

    # plt.figure(1)
    # plt.imshow(imageio.imread(images_filename[i]))
    # plt.figure(2)
    # plt.imshow(imageio.imread(depths_filename[i]))
    # plt.draw()
    # plt.pause(2)
    # # input()

print()
print("invalid_pairs:", invalidPairs)
print(len(invalidPairs))

print("Done")