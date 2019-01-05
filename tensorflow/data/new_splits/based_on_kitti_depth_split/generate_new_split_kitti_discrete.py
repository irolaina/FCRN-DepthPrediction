import collections
import numpy as np

# pylint: disable=line-too-long

# Incoerências
# TODO: Existem entradas duplicadas entre as listas de treinamento e teste do KITTI discrete
# TODO: Ao mesmo tem, existem entradas das duas listas que não existem na lista do KITTI Depth

class Split(object):
    """Create Properties variables"""
    def __init__(self, path):
        self.path = path
        self.filenames = []
        self.pair = []
        self.new_split = []

        self.file = open(self.path, 'r')

class KittiDepth():
    def __init__(self):
        self.train = Split("../../kitti_depth/kitti_depth_train.txt")
        self.test = Split("../../kitti_depth/kitti_depth_val.txt")

    def get_train_filenames(self):
        for i, line in enumerate(self.train.file):
            splitted = line.split()[0].split('/')

            train_filename = '_'.join([splitted[2], splitted[-1]])

            # The Hilbert Maps generated depth maps only for left images (image_02)
            if splitted[3] == 'image_02':
                self.train.filenames.append(train_filename)

            # print(self.train.filename)
            # print(i, splitted) # (85897, ['raw_data', '2011_09_30', '2011_09_30_drive_0028_sync', 'image_03', 'data', '0000001350.png'])

    def get_test_filenames(self):
        for i, line in enumerate(self.test.file):
            splitted = line.split()[0].split('/')

            test_filename = '_'.join([splitted[2], splitted[-1]])

            # The Hilbert Maps generated depth maps only for left images (image_02)
            if splitted[3] == 'image_02':
                self.test.filenames.append(test_filename)

            # print(self.test.filename)
            # print(i, splitted) # 6851 ['raw_data', '2011_09_26', '2011_09_26_drive_0036_sync', 'image_02', 'data', '0000000697.png']

class KittiDiscrete():
    def __init__(self):
        self.train = Split("../../unreliable_splits/kitti_discrete/kitti_discrete_train.txt")
        self.test = Split("../../unreliable_splits/kitti_discrete/kitti_discrete_test.txt")

    def get_train_filenames(self):
        for i, line in enumerate(self.train.file):
            self.train.pair.append(line)
            splitted = line.split()[0].split('/')

            self.train.filenames.append(splitted[-1])
            # print(splitted[-1])
            # print(i, splitted) # (25741, ['2011_09_30', '2011_09_30_drive_0020_sync', 'proc_kitti_nick', 'imgs', '2011_09_30_drive_0020_sync_0000000206.png'])

    def get_test_filenames(self):
        for i, line in enumerate(self.test.file):
            self.test.pair.append(line)
            splitted = line.split()[0].split('/')

            self.test.filenames.append(splitted[-1])
            # print(splitted[-1])
            # print(i, splitted) # 6435 ['2011_09_26', '2011_09_26_drive_0061_sync', 'proc_kitti_nick', 'imgs', '2011_09_26_drive_0061_sync_0000000619.png']

# ====== #
#  Main  #
# ====== #
def main():
    # Reads KITTI Depth and KITTI Discrete files
    print('[Main] Reading KITTI Depth and KITTI Discrete train/test split files...')
    kitti_depth = KittiDepth()
    kitti_discrete = KittiDiscrete()

    # print(kitti_depth.train.path)
    # print(kitti_depth.train.filenames)
    # print(kitti_depth.train.file)
    # print(kitti_depth.test.path)
    # print(kitti_depth.test.filenames)
    # print(kitti_depth.test.file)

    # print(kitti_discrete.train.path)
    # print(kitti_discrete.train.filenames)
    # print(kitti_discrete.train.file)
    # print(kitti_discrete.test.path)
    # print(kitti_discrete.test.filenames)
    # print(kitti_discrete.test.file)

    # Generates filenames for later comparison
    print('[Main] Generating filenames for later comparison...')
    kitti_depth.get_train_filenames()
    kitti_depth.get_test_filenames()

    kitti_discrete.get_train_filenames()
    kitti_discrete.get_test_filenames()

    print()
    print('kitti_depth_train_filenames:', len(kitti_depth.train.filenames))
    print('kitti_depth_test_filenames:', len(kitti_depth.test.filenames))
    print()
    print('kitti_discrete_train_filenames:', len(kitti_discrete.train.filenames))
    print('kitti_discrete_test_filenames:', len(kitti_discrete.test.filenames))
    print()

    kitti_depth.filenames = kitti_depth.train.filenames + kitti_depth.test.filenames
    kitti_discrete.filenames = kitti_discrete.train.filenames + kitti_discrete.test.filenames

    # FIXME: Searches for duplicates

    # np.savetxt('kitti_discrete_train_sorted.txt', np.array(sorted(kitti_discrete.train.filenames)), fmt='%s', delimiter='\t')
    # np.savetxt('kitti_discrete_test_sorted.txt', np.array(sorted(kitti_discrete.test.filenames)), fmt='%s', delimiter='\t')

    print("[Main] Searching for duplicates...")
    try:
        assert len(kitti_discrete.filenames) == len(set(kitti_discrete.filenames)) # Deveriam ser iguais !!!

    except AssertionError:
        duplicates = [item for item, count in collections.Counter(kitti_discrete.filenames).items() if count > 1]

        print("[AssertionError] Existem {} entradas duplicadas nas listas de treinamento e teste do KITTI Discrete!!!".format(len(duplicates)))
        print('{} != {}\n'.format(len(kitti_discrete.filenames), len(set(kitti_discrete.filenames))))

        # Show Duplicates
        print('duplicates:', duplicates)
        print()

    # for pair in kitti_discrete.train.pair:
    #     print(pair)


    # Check which discrete entries are in the kitti depth split.
    # set() removes duplicated filenames.
    print("[Main] Checking which discrete entries are in the kitti depth split.")
    isIn = []
    isNotIn = []
    for i, item in enumerate(set(kitti_discrete.filenames)):
        if item in kitti_depth.train.filenames:
            # print(kitti_discrete.filenames.index(item), item)
            # print(kitti_depth.train.filenames.index(item), kitti_depth.train.filenames[kitti_depth.train.filenames.index(item)])
            kitti_discrete.train.new_split.append(kitti_discrete.train.pair)
            isIn.append(True)

        elif item in kitti_depth.test.filenames:
            # print(kitti_discrete.filenames.index(item), item)
            # print(kitti_depth.test.filenames.index(item), kitti_depth.test.filenames[kitti_depth.test.filenames.index(item)])
            kitti_discrete.test.new_split.append(kitti_discrete.test.pair)
            isIn.append(True)

        else:
            isNotIn.append(item)
            isIn.append(False)

    try:
        assert sum(isIn) == len(isIn)
    except AssertionError:
        print("[AssertionError] Existem {} entradas nas listas de treinamento e test do KITTI Discrete que NÃO existem nas listas do KITTI Depth!!!".format(len(isNotIn)))
        print('{} != {}\n'.format(sum(isIn), len(isIn)))

    print('isNotIn:', isNotIn)
    print()

    # Save New KITTI Discrete lists
    print('# --- KITTI Discrete --- #')
    print("| Old Split\t|\tNew Split|")
    print('| {:9}\t|\t{:9}|'.format(len(kitti_discrete.train.filenames), len(kitti_discrete.train.new_split)))
    print('| {:9}\t|\t{:9}|'.format(len(kitti_discrete.test.filenames), len(kitti_discrete.test.new_split)))
    print('# ---------------------- #\n')

    # Save
    # FIXME: MemoryError
    print("[Main] Saving new split train/test files...")
    np.savetxt('kitti_discrete_train_new_split.txt', np.array(kitti_discrete.train.new_split), fmt='%s', delimiter='')
    np.savetxt('kitti_discrete_test_new_split.txt', np.array(kitti_discrete.test.new_split), fmt='%s', delimiter='')

    kitti_depth.train.file.close()
    kitti_discrete.train.file.close()

    print("Done.")

if __name__ == '__main__':
    main()
