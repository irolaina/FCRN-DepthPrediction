from glob import glob
import numpy as np

# ====== #
#  Test  #
# ====== #
# Read Monodepths Files
eigen_test_kitti_depth_file = open("eigen_test_kitti_depth_files.txt", "r")
eigen_test_kitti_depth_aligned_with_kitti_continuous = []

dataset_path = '/media/nicolas/nicolas_seagate/datasets/kitti/'

# /media/nicolas/nicolas_seagate/datasets/kitti/raw_data/2011_09_26/2011_09_26_drive_0005_sync/proc_kitti_nick/disp2/2011_09_26_drive_0005_sync_0000000006.png
# /media/nicolas/nicolas_seagate/datasets/kitti/depth/depth_prediction/data/val/2011_09_26_drive_0020_sync/proj_depth/groundtruth/image_02/0000000005.png
count = 0
for line in eigen_test_kitti_depth_file:
    splitted = line.split()[1].split('/')
    splitted.insert(0, 'raw_data')
    splitted.pop(1)
    splitted.pop(1)
    splitted.pop(1)
    splitted.pop(1)
    splitted.pop(-2)
    splitted.pop(-2)
    splitted.pop(-2)
    splitted.insert(1, splitted[1].split('_drive')[0])
    splitted.insert(3, 'proc_kitti_nick/disp2')
    splitted[-1] = splitted[-3] + '_' + splitted[-1]

    print(dataset_path+('/'.join(splitted)))
    depth_continuous = glob(dataset_path+('/'.join(splitted)))

    splitted = line.split()[1].split('/')
    # splitted.insert(0, 'raw_data')
    semi_dense = glob(dataset_path + ('/'.join(splitted)))

    if semi_dense and depth_continuous:
        newline = [depth_continuous[0].replace(dataset_path, ''), semi_dense[0].replace(dataset_path, '')]
        eigen_test_kitti_depth_aligned_with_kitti_continuous.append(newline)
        count += 1

    # Display
    print(splitted)
    # print(dataset_path + ('/'.join(splitted)))
    # print('/'.join(splitted))
    # print(count, line.split()[0])
    print("semi_dense: {}".format(semi_dense))
    print("depth_continuous: {}".format(depth_continuous))
    print()

eigen_test_kitti_depth_file.close()

# Display Results
for i, pair in enumerate(eigen_test_kitti_depth_aligned_with_kitti_continuous):
    print(i, pair)

print("\n{} pairs were found!".format(count))
print()

# Save
np.savetxt('eigen_test_kitti_depth_aligned_with_kitti_continuous_files.txt', np.array(eigen_test_kitti_depth_aligned_with_kitti_continuous), fmt='%s', delimiter='\t')

print("\nDone.")
