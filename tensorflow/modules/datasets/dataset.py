# ===========
#  Libraries
# ===========
from ..size import Size
from ..filenames import FilenamesHandler
from ..args import args


# ===================
#  Class Declaration
# ===================
class Dataset(FilenamesHandler):
    def __init__(self, *args, **kwargs):
        self.dataset_path = kwargs.pop('dataset_path')
        self.name = kwargs.pop('name')
        height = kwargs.pop('height')
        width = kwargs.pop('width')
        self.max_depth = kwargs.pop('max_depth')  # Max Depth to limit predictions

        super(Dataset, self).__init__(*args, **kwargs)

        self.image_size = Size(height, width, 3)
        self.depth_size = Size(height, width, 1)

        # Train/Test Split Ratio
        self.ratio = 0.8

        print("[Dataloader] %s object created." % self.name)

    def get_file_path(self, mode, test_split, test_file_path):
        if test_split == 'eigen' or test_split == 'eigen_continuous':  # FIXME: O mais correto seria chamar de 'eigen_stereo' e 'eigen_stereo_continuous'
            file_path = 'modules/third_party/monodepth/utils/filenames/eigen_test_files.txt'

            # Overwrite the 'dataset_path' specified by the dataset
            self.dataset_path = '/media/nicolas/nicolas_seagate/datasets/kitti/raw_data/'

        elif test_split == 'kitti_stereo':
            file_path = 'modules/third_party/monodepth/utils/filenames/kitti_stereo_2015_test_files.txt'

            # Overwrite the 'dataset_path' specified by the dataset
            self.dataset_path = '/media/nicolas/nicolas_seagate/datasets/kitti/stereo/stereo2015/data_scene_flow/'

        elif test_split == 'eigen_kitti_depth':  # FIXME: Validar!
            file_path = 'data/new_splits/kitti_splits_based_on_monodepth_files/kitti_depth/eigen_test_files.txt'

            # Overwrite the 'dataset_path' specified by the dataset
            self.dataset_path = '/media/nicolas/nicolas_seagate/datasets/kitti/'

        else:
            if test_file_path == '':
                file_path = 'data/' + self.name + '_' + mode + '.txt'
            else:
                file_path = test_file_path

        args.test_file_path = file_path

        return file_path
