from glob import glob
import numpy as np

# ====== #
#  Test  #
# ====== #
# Read Monodepths Files
eigen_test_file = open("../../../modules/third_party/monodepth/utils/filenames/eigen_test_files.txt", "r")
eigen_test = []

dataset_path = '/media/nicolas/nicolas_seagate/datasets/kitti/'

count = 0
for line in eigen_test_file:
    splitted = line.split()[0].split('/')
    splitted.insert(0, 'depth/depth_prediction/data/*')
    splitted.insert(3, 'proj_depth/groundtruth')
    splitted.pop(-2)
    splitted.pop(1)

    depth = glob(dataset_path+('/'.join(splitted)))

    splitted = line.split()[0].split('/')
    splitted.insert(0, 'raw_data')

    image = glob(dataset_path + ('/'.join(splitted)))

    if image and depth: # TODO: change to 'image and depth
        newline = [image[0].replace(dataset_path, ''), depth[0].replace(dataset_path, '')]
        eigen_test.append(newline)
        count += 1


    # Display
    # print(splitted)
    # print(dataset_path + ('/'.join(splitted)))
    # print('/'.join(splitted))
    print(count, line.split()[0])
    print("image: {}".format(image))
    print("depth: {}".format(depth))
    print()

eigen_test_file.close()

# Display Results
for i, pair in enumerate(eigen_test):
    print(i, pair)

print("\n{} pairs were found!".format(count))
print()

# Save
np.savetxt('kitti_depth_eigen_test_files.txt', np.array(eigen_test), fmt='%s', delimiter='\t')

print("Done.")