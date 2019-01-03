import numpy as np
import collections

# Incoerências
# TODO: Existem entradas duplicadas entre as listas de treinamento e teste do KITTI discrete
# TODO: Ao mesmo tem, existem entradas das duas listas que não existem na lista do KITTI Depth

# ======= #
#  Train  #
# ======= #
# Read Kitti Depth File
kitti_depth_train_file = open("../../kitti_depth/kitti_depth_train.txt", "r")
kitti_depth_train_filenames = []

kitti_depth_test_file = open("../../kitti_depth/kitti_depth_val.txt", "r")
kitti_depth_test_filenames = []

kitti_discrete_train_file = open("../../unreliable_splits/kitti_discrete/kitti_discrete_train.txt") # Old
kitti_discrete_train_filenames = []
kitti_discrete_train_pair = []

kitti_discrete_test_file = open("../../unreliable_splits/kitti_discrete/kitti_discrete_test.txt") # Old
kitti_discrete_test_filenames = []
kitti_discrete_test_pair = []

# Generating filenames for later comparison
for i, line in enumerate(kitti_depth_train_file):
    splitted = line.split()[0].split('/')

    kitti_depth_train_filename = '_'.join([splitted[2], splitted[-1]])

    # The Hilbert Maps generated depth maps only for left images (image_02)
    if splitted[3] == 'image_02':
        kitti_depth_train_filenames.append(kitti_depth_train_filename)

    # print(kitti_depth_train_filename)
    # print(i, splitted) # (85897, ['raw_data', '2011_09_30', '2011_09_30_drive_0028_sync', 'image_03', 'data', '0000001350.png'])

for i, line in enumerate(kitti_depth_test_file):
    splitted = line.split()[0].split('/')

    kitti_depth_test_filename = '_'.join([splitted[2], splitted[-1]])

    # The Hilbert Maps generated depth maps only for left images (image_02)
    if splitted[3] == 'image_02':
        kitti_depth_test_filenames.append(kitti_depth_test_filename)

    # print(kitti_depth_test_filename)
    # print(i, splitted) # 6851 ['raw_data', '2011_09_26', '2011_09_26_drive_0036_sync', 'image_02', 'data', '0000000697.png']

for i, line in enumerate(kitti_discrete_train_file):
    kitti_discrete_train_pair.append(line)
    splitted = line.split()[0].split('/')

    kitti_discrete_train_filenames.append(splitted[-1])
    # print(splitted[-1])
    # print(i, splitted) # (25741, ['2011_09_30', '2011_09_30_drive_0020_sync', 'proc_kitti_nick', 'imgs', '2011_09_30_drive_0020_sync_0000000206.png'])

for i, line in enumerate(kitti_discrete_test_file):
    kitti_discrete_test_pair.append(line)
    splitted = line.split()[0].split('/')

    kitti_discrete_test_filenames.append(splitted[-1])
    # print(splitted[-1])
    # print(i, splitted) # 6435 ['2011_09_26', '2011_09_26_drive_0061_sync', 'proc_kitti_nick', 'imgs', '2011_09_26_drive_0061_sync_0000000619.png']

print()
print('kitti_depth_train_filenames:', len(kitti_depth_train_filenames))
print()
print('kitti_discrete_train_filenames:', len(kitti_discrete_train_filenames))
print('kitti_discrete_test_filenames:', len(kitti_discrete_test_filenames))
print()

kitti_depth_filenames = kitti_depth_train_filenames + kitti_depth_test_filenames
kitti_discrete_filenames = kitti_discrete_train_filenames + kitti_discrete_test_filenames

# Search for duplicates
np.savetxt('kitti_discrete_train_sorted.txt', np.array(sorted(kitti_discrete_train_filenames)), fmt='%s', delimiter='\t')
np.savetxt('kitti_discrete_test_sorted.txt', np.array(sorted(kitti_discrete_test_filenames)), fmt='%s', delimiter='\t')

# FIXME:
try:
    assert len(kitti_discrete_filenames) == len(set(kitti_discrete_filenames)) # Deveriam ser iguais !!!

except AssertionError:
    # Show Duplicates
    print()
    print("Existem entradas duplicadas nas listas de treinamento e teste do KITTI Discrete!!!")
    print('{} != {}\n'.format(len(kitti_discrete_filenames), len(set(kitti_discrete_filenames))))

    duplicates = [item for item, count in collections.Counter(kitti_discrete_filenames).items() if count > 1]
    print('duplicates:', duplicates)
    print(len(duplicates))

# Check which discrete entries are in the kitti depth split.
# set() removes duplicated filenames.
isIn = []
kitti_discrete_train_new = []
kitti_discrete_test_new = []

for i, item in enumerate(set(kitti_discrete_filenames)):
    if item in kitti_depth_train_filenames:
        isIn.append(True)
        kitti_discrete_train_new.append(kitti_discrete_train_pair[i]) # FIXME: Recuperar os paths do par
    elif item in kitti_depth_test_filenames:
        isIn.append(True)
        kitti_discrete_test_new.append(kitti_discrete_test_pair[i]) # FIXME: Recuperar os paths do par
    else:
        isIn.append(False)

try:
    assert sum(isIn) == len(isIn)
except AssertionError:
    print()
    print("Existem entradas das listas de treinamento e test do KITTI Discrete que NÃO existem nas listas do KITTI Depth!!!\n")

print(isIn)
print('{}/{}'.format(sum(isIn),len(isIn)))

# TODO: Terminar
# Save New KITTI Discrete lists
print(len(kitti_discrete_train_new))
print(len(kitti_discrete_test_new))

# TODO: Terminar
# Save
# np.savetxt('kitti_discrete_train_new.txt', np.array(sorted(kitti_discrete_train_filenames)), fmt='%s', delimiter='\t')
# np.savetxt('kitti_discrete_test_new.txt', np.array(sorted(kitti_discrete_test_filenames)), fmt='%s', delimiter='\t')
