import numpy as np

# ======= #
#  Train  #
# ======= #
# Read Monodepths Files
eigen_train_file = open("../../../modules/third_party/monodepth/utils/filenames/eigen_train_files.txt", "r")
eigen_train = []

for line in eigen_train_file:
    splitted = line.split()[0].split('/')
    splitted.insert(0, 'raw_data')
    splitted[-1] = splitted[-1].replace('.jpg', '.png')
    image = '/'.join(splitted)

    depth = splitted
    depth[0] = depth[0].replace('raw_data', 'depth')
    depth[1] = 'depth_prediction/data/train'
    depth.insert(3, 'proj_depth/groundtruth')
    depth.pop(-2)
    depth = '/'.join(depth)

    newline = [image, depth]

    # print(image)
    # print(depth)
    # print(newline)
    # print(splitted)

    eigen_train.append(newline)

# Display
for i, pair in enumerate(eigen_train):
    print(i, pair)

# Save
np.savetxt('kitti_depth/eigen_train_files.txt', np.array(eigen_train), fmt='%s', delimiter='\t')
eigen_train_file.close()

# TODO: Faz sentido aplicar o eigen split ao KITTI Depth? Agora, o KITTI Depth possui um próprio conjunto de imagens de validação (Teste, em minha terminologia)
# ====== #
#  Test  #
# ====== #
# Read Monodepths Files
eigen_test_file = open("../../../modules/third_party/monodepth/utils/filenames/eigen_test_files.txt", "r")
eigen_test = []

for line in eigen_test_file:
    splitted = line.split()[0].split('/')
    splitted.insert(0, 'raw_data')
    splitted[-1] = splitted[-1].replace('.jpg', '.png')
    image = '/'.join(splitted)

    depth = splitted
    depth[0] = depth[0].replace('raw_data', 'depth')
    depth[1] = 'depth_prediction/data/val'  # TODO: Isto está certo?
    depth.insert(3, 'proj_depth/groundtruth')
    depth.pop(-2)
    depth = '/'.join(depth)

    newline = [image, depth]

    # print(image)
    # print(depth)
    # print(newline)
    # print(splitted)

    eigen_test.append(newline)  # TODO: Descomentar

# Display
for i, pair in enumerate(eigen_test):
    print(i, pair)

# Save
np.savetxt('kitti_depth/eigen_test_files.txt', np.array(eigen_test), fmt='%s', delimiter='\t')
eigen_test_file.close()

print("\nDone.")