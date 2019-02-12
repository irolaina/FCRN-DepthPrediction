import collections
import glob
import os
import time
import numpy as np
import sys


# pylint: disable=line-too-long

# Incoerências
# TODO: Ao mesmo tem, existem entradas das duas listas que não existem na lista do KITTI Depth
# TODO: Número de imagens encontradas no KITTI Continuous do KITTI Discrete é diferente!!!


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

        self.pairs = []
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


class KittiDiscrete:
    def __init__(self):
        self.train = Split("../../unreliable_splits/kitti_discrete/kitti_discrete_train.txt")
        self.test = Split("../../unreliable_splits/kitti_discrete/kitti_discrete_test.txt")

        self.filenames = []

    def get_filenames(self):
        for i, line in enumerate(self.pairs):
            splitted = line[0].split('/')

            self.filenames.append(splitted[-1])
            # print(splitted[-1])
            # print(i, splitted) # (25741, ['2011_09_30', '2011_09_30_drive_0020_sync', 'proc_kitti_nick', 'imgs', '2011_09_30_drive_0020_sync_0000000206.png'])

    @staticmethod
    def read_text_file(filename):
        print("\n[Dataloader] Loading '%s'..." % filename)
        try:
            data = np.genfromtxt(filename, dtype='str', delimiter='\t')
            # print(data.shape)
        except OSError:
            print("[OSError] Could not find the '%s' file." % filename)
            sys.exit()

        # # Parsing Data
        # image_filenames = list(data[:, 0])
        # depth_filenames = list(data[:, 1])

        # return image_filenames, depth_filenames

        return list(data)

    def get_image_depth_pair_relative_paths(self):
        file_path = 'kitti_discrete.txt'

        if os.path.exists(file_path):
            self.pairs = self.read_text_file(file_path)
        else:
            print("[Dataloader] '%s' doesn't exist..." % file_path)
            print("[Dataloader] Searching files using glob (This may take a while)...")

            dataset_root = "/media/nicolas/nicolas_seagate/datasets/"
            self.dataset_path = dataset_root + "kitti/raw_data/"

            # Finds input images and labels inside the list of folders.
            image_filenames_tmp = glob.glob(self.dataset_path + "2011_*/*/proc_kitti_nick/imgs/*.png")
            depth_filenames_tmp = glob.glob(self.dataset_path + "2011_*/*/proc_kitti_nick/disp1/*.png")

            image_filenames_aux = [os.path.splitext(os.path.split(image)[1])[0] for image in image_filenames_tmp]
            depth_filenames_aux = [os.path.splitext(os.path.split(depth)[1])[0] for depth in depth_filenames_tmp]

            # TODO: Add Comment
            image_filenames, depth_filenames, n2, m2 = self.search_pairs(image_filenames_tmp, depth_filenames_tmp,
                                                                         image_filenames_aux, depth_filenames_aux)

            # Removes the prefix '/media/nicolas/nicolas_seagate/datasets/kitti/raw_data/'
            self.pairs = [(image_filename.replace('/media/nicolas/nicolas_seagate/datasets/kitti/raw_data/', ''),
                           depth_filename.replace('/media/nicolas/nicolas_seagate/datasets/kitti/raw_data/', '')) for
                          (image_filename, depth_filename) in zip(image_filenames, depth_filenames)]
            # for i in filenames:
            #     print(i)
            # input("enter")

            np.savetxt('kitti_discrete.txt', self.pairs, fmt='%s', delimiter='\t')

    @staticmethod
    def search_pairs(image_filenames_tmp, depth_filenames_tmp,
                     image_filenames_aux,
                     depth_filenames_aux):  # TODO: Preciso realmente ter essas duas variaveis? Podem ser unificadas?
        image_filenames = []
        depth_filenames = []

        # print(image_filenames_tmp)
        # print(len(image_filenames_tmp))
        # input("image_filenames_tmp")
        # print(depth_filenames_tmp)
        # print(len(depth_filenames_tmp))
        # input("depth_filenames_tmp")

        # print(image_filenames_aux)
        # print(len(image_filenames_aux))
        # input("image_filenames_aux")
        # print(depth_filenames_aux)
        # print(len(depth_filenames_aux))
        # input("depth_filenames_aux")

        _, m = len(image_filenames_aux), len(depth_filenames_aux)

        # Sequential Search. This kind of search ensures that the images are paired!
        print("[Dataloader] Checking if RGB and Depth images are paired... ")

        start = time.time()
        for j, depth in enumerate(depth_filenames_aux):
            print("%d/%d" % (j + 1, m))  # Debug
            for i, image in enumerate(image_filenames_aux):
                if image == depth:
                    image_filenames.append(image_filenames_tmp[i])
                    depth_filenames.append(depth_filenames_tmp[j])

        n2, m2 = len(image_filenames), len(depth_filenames)
        if not n2 == m2:
            print("[AssertionError] Length must be equal!")
            raise AssertionError()
        print("time: %f s" % (time.time() - start))

        # Shuffles
        s = np.random.choice(n2, n2, replace=False)
        image_filenames = list(np.array(image_filenames)[s])
        depth_filenames = list(np.array(depth_filenames)[s])

        return image_filenames, depth_filenames, n2, m2


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

    kitti_discrete.get_image_depth_pair_relative_paths()
    kitti_discrete.get_filenames()

    print()
    print('kitti_depth_train_filenames:', len(kitti_depth.train.filenames))
    print('kitti_depth_test_filenames:', len(kitti_depth.test.filenames))
    print()
    print('kitti_discrete_pairs:', len(kitti_discrete.pairs))
    print('kitti_discrete_filenames:', len(kitti_discrete.filenames))
    print()

    # TODO: Melhorar isto
    kitti_depth.filenames = kitti_depth.train.filenames + kitti_depth.test.filenames

    np.savetxt('kitti_discrete_sorted.txt', sorted(kitti_discrete.filenames), fmt='%s', delimiter='\t')

    print("[Main] Searching for duplicates...")
    try:
        # Devem ser iguais !!!
        if len(kitti_discrete.filenames) != len(set(kitti_discrete.filenames)):
            raise AssertionError
        else:
            print('{} == {}\n'.format(len(kitti_discrete.filenames), len(set(kitti_discrete.filenames))))

    except AssertionError:
        duplicates = [item for item, count in collections.Counter(kitti_discrete.filenames).items() if count > 1]

        print(
            "[AssertionError] Existem {} entradas duplicadas nas listas de treinamento e teste do KITTI Discrete!!!".format(
                len(duplicates)))
        print('{} != {}\n'.format(len(kitti_discrete.filenames), len(set(kitti_discrete.filenames))))

        # Show Duplicates
        print('duplicates:', duplicates)
        print()

    # Check which discrete entries are in the kitti depth split.
    # set() removes duplicated filenames.
    print("[Main] Checking which KITTI Discrete entries are in the KITTI Depth split.")
    is_in = []
    is_not_in = []

    # Transforms list of np.array of strings, to list of strings.
    kitti_discrete.pairs = [pair[0] + '\t' + pair[1] for pair in kitti_discrete.pairs]

    for item in set(kitti_discrete.filenames):
        if item in kitti_depth.train.filenames:
            # print(kitti_discrete.filenames.index(item), item)
            # print(kitti_depth.train.filenames.index(item), kitti_depth.train.filenames[kitti_depth.train.filenames.index(item)])

            # Find correspondent pair for the queried item.
            pair = [s for s in kitti_discrete.pairs if item in s]
            # print(pair)

            kitti_discrete.train.new_split.append(pair)
            is_in.append(True)

        elif item in kitti_depth.test.filenames:
            # print(kitti_discrete.filenames.index(item), item)
            # print(kitti_depth.test.filenames.index(item), kitti_depth.test.filenames[kitti_depth.test.filenames.index(item)])

            # Find correspondent pair for the queried item.
            pair = [s for s in kitti_discrete.pairs if item in s]
            # print(pair)

            kitti_discrete.test.new_split.append(pair)
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
            "[AssertionError] Existem {} entradas nas listas de treinamento e test do KITTI Discrete que NÃO existem nas listas do KITTI Depth!!!".format(
                len(is_not_in)))
        print('{} != {}\n'.format(sum(is_in), len(is_in)))

    print('is_not_in:', is_not_in)
    print()

    # Save New KITTI Discrete lists
    print('# --- KITTI Discrete --- #')
    print("| \tPairs\t|\tNew Split|")
    print('| {:9}\t|\t{:9}|'.format(len(kitti_discrete.filenames), len(kitti_discrete.train.new_split)))
    print('| {:9}\t|\t{:9}|'.format('', len(kitti_discrete.test.new_split)))
    print('# ---------------------- #\n')

    # Save
    print("[Main] Saving new split train/test files...")

    np.savetxt('kitti_discrete_train_new_split.txt', kitti_discrete.train.new_split, fmt='%s', delimiter='')
    np.savetxt('kitti_discrete_test_new_split.txt', kitti_discrete.test.new_split, fmt='%s', delimiter='')

    kitti_depth.train.file.close()
    kitti_depth.test.file.close()

    kitti_discrete.train.file.close()

    print("Done.")


if __name__ == '__main__':
    main()
