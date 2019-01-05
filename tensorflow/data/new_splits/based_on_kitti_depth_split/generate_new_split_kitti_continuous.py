import collections
import numpy as np

# pylint: disable=line-too-long

# Incoerências
# TODO: Existem entradas duplicadas entre as listas de treinamento e teste do KITTI continuous
# TODO: Ao mesmo tem, existem entradas das duas listas que não existem na lista do KITTI Depth
# TODO: Número de imagens encontradas no KITTI Continuous do KITTI Continuous é diferente!!!


class Split:
    """Create Properties variables"""

    def __init__(self, path):
        self.path = path
        self.filenames = []
        self.pair = []
        self.new_split = []

        self.file = open(self.path, 'r')


class KittiDepth:
    def __init__(self):
        self.train = Split("../../kitti_depth/kitti_depth_train.txt")
        self.test = Split("../../kitti_depth/kitti_depth_val.txt")

        self.filenames = []

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


class KittiContinuous:
    def __init__(self):
        self.train = Split("../../unreliable_splits/kitti_continuous/kitti_continuous_train.txt")
        self.test = Split("../../unreliable_splits/kitti_continuous/kitti_continuous_test.txt")

        self.filenames = []

    def get_train_filenames(self):
        for i, line in enumerate(self.train.file):
            self.train.pair.append(line.rstrip())
            splitted = line.split()[0].split('/')

            self.train.filenames.append(splitted[-1])
            # print(splitted[-1])
            # print(i, splitted) # (25741, ['2011_09_30', '2011_09_30_drive_0020_sync', 'proc_kitti_nick', 'imgs', '2011_09_30_drive_0020_sync_0000000206.png'])

    def get_test_filenames(self):
        for i, line in enumerate(self.test.file):
            self.test.pair.append(line.rstrip())
            splitted = line.split()[0].split('/')

            self.test.filenames.append(splitted[-1])
            # print(splitted[-1])
            # print(i, splitted) # 6435 ['2011_09_26', '2011_09_26_drive_0061_sync', 'proc_kitti_nick', 'imgs', '2011_09_26_drive_0061_sync_0000000619.png']


# ====== #
#  Main  #
# ====== #
def main():
    # Reads KITTI Depth and KITTI Continuous files
    print('[Main] Reading KITTI Depth and KITTI Continuous train/test split files...')
    kitti_depth = KittiDepth()
    kitti_continuous = KittiContinuous()

    # print(kitti_depth.train.path)
    # print(kitti_depth.train.filenames)
    # print(kitti_depth.train.file)
    # print(kitti_depth.test.path)
    # print(kitti_depth.test.filenames)
    # print(kitti_depth.test.file)

    # print(kitti_continuous.train.path)
    # print(kitti_continuous.train.filenames)
    # print(kitti_continuous.train.file)
    # print(kitti_continuous.test.path)
    # print(kitti_continuous.test.filenames)
    # print(kitti_continuous.test.file)

    # Generates filenames for later comparison
    print('[Main] Generating filenames for later comparison...')
    kitti_depth.get_train_filenames()
    kitti_depth.get_test_filenames()

    kitti_continuous.get_train_filenames()
    kitti_continuous.get_test_filenames()

    print()
    print('kitti_depth_train_filenames:', len(kitti_depth.train.filenames))
    print('kitti_depth_test_filenames:', len(kitti_depth.test.filenames))
    print()
    print('kitti_continuous_train_filenames:', len(kitti_continuous.train.filenames))
    print('kitti_continuous_test_filenames:', len(kitti_continuous.test.filenames))
    print()

    # TODO: Melhorar isto
    kitti_depth.filenames = kitti_depth.train.filenames + kitti_depth.test.filenames
    kitti_continuous.filenames = kitti_continuous.train.filenames + kitti_continuous.test.filenames
    kitti_continuous.pairs = kitti_continuous.train.pair + kitti_continuous.test.pair

    # np.savetxt('kitti_continuous_train_sorted.txt', np.array(sorted(kitti_continuous.train.filenames)), fmt='%s', delimiter='\t')
    # np.savetxt('kitti_continuous_test_sorted.txt', np.array(sorted(kitti_continuous.test.filenames)), fmt='%s', delimiter='\t')

    print("[Main] Searching for duplicates...")
    try:
        # Devem ser iguais !!!
        if len(kitti_continuous.filenames) != len(set(kitti_continuous.filenames)):
            raise AssertionError

    except AssertionError:
        duplicates = [item for item, count in collections.Counter(kitti_continuous.filenames).items() if count > 1]

        print(
            "[AssertionError] Existem {} entradas duplicadas nas listas de treinamento e teste do KITTI Continuous!!!".format(
                len(duplicates)))
        print('{} != {}\n'.format(len(kitti_continuous.filenames), len(set(kitti_continuous.filenames))))

        # Show Duplicates
        print('duplicates:', duplicates)
        print()

    # for pair in kitti_continuous.train.pair:
    #     print(pair)

    # Check which continuous entries are in the kitti depth split.
    # set() removes duplicated filenames.
    print("[Main] Checking which KITTI Continuous entries are in the KITTI Depth split.")
    is_in = []
    is_not_in = []
    for item in set(kitti_continuous.filenames):
        if item in kitti_depth.train.filenames:
            # print(kitti_continuous.filenames.index(item), item)
            # print(kitti_depth.train.filenames.index(item), kitti_depth.train.filenames[kitti_depth.train.filenames.index(item)])

            # Find correspondent pair for the queried item.
            pair = [s for s in kitti_continuous.pairs if item in s][0]
            # print(pair)

            kitti_continuous.train.new_split.append(pair)
            is_in.append(True)

        elif item in kitti_depth.test.filenames:
            # print(kitti_continuous.filenames.index(item), item)
            # print(kitti_depth.test.filenames.index(item), kitti_depth.test.filenames[kitti_depth.test.filenames.index(item)])

            # Find correspondent pair for the queried item.
            pair = [s for s in kitti_continuous.pairs if item in s][0]
            # print(pair)

            kitti_continuous.test.new_split.append(pair)
            is_in.append(True)

        else:
            is_not_in.append(item)
            is_in.append(False)

    try:
        # Devem ser iguais !!!
        if sum(is_in) != len(is_in):
            raise AssertionError
    except AssertionError:
        print(
            "[AssertionError] Existem {} entradas nas listas de treinamento e test do KITTI Continuous que NÃO existem nas listas do KITTI Depth!!!".format(
                len(is_not_in)))
        print('{} != {}\n'.format(sum(is_in), len(is_in)))

    print('is_not_in:', is_not_in)
    print()

    # Save New KITTI Continuous lists
    print('# --- KITTI Continuous --- #')
    print("| Old Split\t|\tNew Split|")
    print('| {:9}\t|\t{:9}|'.format(len(kitti_continuous.train.filenames), len(kitti_continuous.train.new_split)))
    print('| {:9}\t|\t{:9}|'.format(len(kitti_continuous.test.filenames), len(kitti_continuous.test.new_split)))
    print('# ---------------------- #\n')

    # Save
    print("[Main] Saving new split train/test files...")

    np.savetxt('kitti_continuous_train_new_split.txt', kitti_continuous.train.new_split, fmt='%s', delimiter='')
    np.savetxt('kitti_continuous_test_new_split.txt', kitti_continuous.test.new_split, fmt='%s', delimiter='')

    kitti_depth.train.file.close()
    kitti_continuous.train.file.close()

    print("Done.")


if __name__ == '__main__':
    main()
