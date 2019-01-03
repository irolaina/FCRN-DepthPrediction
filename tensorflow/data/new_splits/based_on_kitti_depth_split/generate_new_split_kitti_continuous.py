# import numpy as np
#
# # ======= #
# #  Train  #
# # ======= #
# # Read Monodepths Files
# eigen_train_file = open("../monodepth/eigen_train_files.txt", "r")
# eigen_train = []
#
# for line in eigen_train_file:
#     splitted = line.split()[0].split('/')
#
#     # image_02 -> proc_kitti_nick
#     splitted[2] = 'proc_kitti_nick'
#
#     # data -> imgs
#     splitted[3] = 'imgs'
#
#     # 0000000077.jpg -> 2011_09_26_drive_0051_sync_0000000077.jpg
#     splitted[-1] = '_'.join([splitted[1], splitted[-1]])
#
#     # .jpg to .png
#     splitted[-1] = splitted[-1].replace('.jpg', '.png')
#
#     # Join Everything
#     newline = '/'.join(splitted)
#
#     # print(newline)
#     # print(splitted[1])
#     # print(splitted)
#
#     eigen_train.append([newline, newline.replace('imgs', 'disp1')])
#
# # Display
# for i, pair in enumerate(eigen_train):
#      print(i, pair)
#
# # Save
# np.savetxt('kittidiscrete/eigen_train_files.txt', np.array(eigen_train), fmt='%s', delimiter='\t')
# eigen_train_file.close()
#
# # ====== #
# #  Test  #
# # ====== #
# # Read Monodepths Files
# eigen_test_file = open("../monodepth/eigen_test_files.txt", "r")
# eigen_test = []
#
# for line in eigen_test_file:
#     splitted = line.split()[0].split('/')
#
#     # image_02 -> proc_kitti_nick
#     splitted[2] = 'proc_kitti_nick'
#
#     # data -> imgs
#     splitted[3] = 'imgs'
#
#     # 0000000077.jpg -> 2011_09_26_drive_0051_sync_0000000077.jpg
#     splitted[-1] = '_'.join([splitted[1], splitted[-1]])
#
#     # .jpg to .png
#     splitted[-1] = splitted[-1].replace('.jpg', '.png')
#
#     # Join Everything
#     newline = '/'.join(splitted)
#
#     # print(newline)
#     # print(splitted[1])
#     # print(splitted)
#
#     eigen_test.append([newline, newline.replace('imgs', 'disp1')])
#
# # Display
# for i, pair in enumerate(eigen_test):
#      print(i, pair)
#
# # Save
# np.savetxt('kittidiscrete/eigen_test_files.txt', np.array(eigen_test), fmt='%s', delimiter='\t')
# eigen_test_file.close()
