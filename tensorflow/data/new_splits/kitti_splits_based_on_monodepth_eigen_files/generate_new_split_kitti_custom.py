import numpy as np

# select_dataset = 'kitti_discrete'
select_dataset = 'kitti_continuous'

if select_dataset == 'kitti_discrete':
    disp_folder = 'disp1'
elif select_dataset == 'kitti_continuous':
    disp_folder = 'disp2'
else:
    raise SystemError

# ======= #
#  Train  #
# ======= #
# Read Monodepths Files
eigen_train_file = open("../../../modules/third_party/monodepth/utils/filenames/eigen_train_files.txt", "r")
eigen_train = []

for line in eigen_train_file:
    splitted = line.split()[0].split('/')

    # image_02 -> proc_kitti_nick
    splitted[2] = 'proc_kitti_nick'

    # data -> imgs
    splitted[3] = 'imgs'

    # 0000000077.jpg -> 2011_09_26_drive_0051_sync_0000000077.jpg
    splitted[-1] = '_'.join([splitted[1], splitted[-1]])

    # .jpg to .png
    splitted[-1] = splitted[-1].replace('.jpg', '.png')

    # Join Everything
    newline = '/'.join(splitted)

    # print(newline)
    # print(splitted[1])
    # print(splitted)

    eigen_train.append([newline, newline.replace('imgs', disp_folder)])

# Display
for i, pair in enumerate(eigen_train):
    print(i, pair)

# Save
np.savetxt(select_dataset + '/eigen_train_files.txt', np.array(eigen_train), fmt='%s', delimiter='\t')
eigen_train_file.close()

# ====== #
#  Test  #
# ====== #
# Read Monodepths Files
eigen_test_file = open("../../../modules/third_party/monodepth/utils/filenames/eigen_test_files.txt", "r")
eigen_test = []

for line in eigen_test_file:
    splitted = line.split()[0].split('/')

    # image_02 -> proc_kitti_nick
    splitted[2] = 'proc_kitti_nick'

    # data -> imgs
    splitted[3] = 'imgs'

    # 0000000077.jpg -> 2011_09_26_drive_0051_sync_0000000077.jpg
    splitted[-1] = '_'.join([splitted[1], splitted[-1]])

    # .jpg to .png
    splitted[-1] = splitted[-1].replace('.jpg', '.png')

    # Join Everything
    newline = '/'.join(splitted)

    # print(newline)
    # print(splitted[1])
    # print(splitted)

    eigen_test.append([newline, newline.replace('imgs', disp_folder)])

# Display
for i, pair in enumerate(eigen_test):
    print(i, pair)

# Save
np.savetxt(select_dataset + '/eigen_test_files.txt', np.array(eigen_test), fmt='%s', delimiter='\t')
eigen_test_file.close()

print("\nDone.")
