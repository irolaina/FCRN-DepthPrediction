import numpy as np
import time, sys
import imageio
from tqdm import tqdm

def join_dataset_path(filenames, dataset_path):
    timer = -time.time()
    filenames = [dataset_path + filename for filename in filenames]
    timer += time.time()
    print('time:', timer, 's\n')

    return filenames

def read_text_file(filename, dataset_path):
    print("\n[Dataloader] Loading '%s'..." % filename)
    try:
        data = np.genfromtxt(filename, dtype='str', delimiter='\t')
        # print(data.shape)
    except OSError:
        print("[OSError] Could not find the '%s' file." % filename)
        raise SystemExit


    # Parsing Data
    image_filenames = list(data[:, 0])
    depth_filenames = list(data[:, 1])

    image_filenames = join_dataset_path(image_filenames, dataset_path)
    depth_filenames = join_dataset_path(depth_filenames, dataset_path)

    return image_filenames, depth_filenames

image_filenames, depth_filenames = read_text_file('tmp/kitti_depth_train.txt'     , '/media/nicolas/nicolas_seagate/datasets/kitti/'         )  # KITTI Depth
# image_filenames, depth_filenames = read_text_file('tmp/kitti_discrete_train.txt'  , '/media/nicolas/nicolas_seagate/datasets/kitti/raw_data/')  # KITTI Discrete
# image_filenames, depth_filenames = read_text_file('tmp/kitticontinuous_train.txt', '/media/nicolas/nicolas_seagate/datasets/kitti/raw_data/')  # KITTI Continuous

# Print Arrays Content
# for image in image_filenames:
#     print(image)
# print(len(image_filenames))
# for depth in depth_filenames:
#     print(depth)
# print(len(depth_filenames))

array_num_valid_pixels = []

sum = 0
n = 1

for depth_path in tqdm(depth_filenames):
    depth = imageio.imread(depth_path)
    idx = np.where(depth>0)

    num_valid_pixels = len(idx[0])

    sum += num_valid_pixels
    mean = int(sum/n)
    n += 1

    # print(depth_path)
    # print(idx)
    # print(num_valid_pixels, int(mean))

print("mean: ", mean)
print('Done.')