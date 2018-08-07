# ===========
#  Libraries
# ===========
from ..size import Size
from ..filenames import FilenamesHandler


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

    def get_file_path(self, mode, test_split, test_file_path):  # TODO: change name
        if test_split == 'eigen' or test_split == 'eigen_continuous':
            file = 'modules/third_party/monodepth/utils/filenames/eigen_test_files.txt'

            # Overwrite the 'dataset_path' specified by the dataset
            self.dataset_path = '/media/nicolas/nicolas_seagate/datasets/kitti/raw_data/'
        elif test_split == 'kitti':
            file = 'modules/third_party/monodepth/utils/filenames/kitti_stereo_2015_test_files.txt'

            # Overwrite the 'dataset_path' specified by the dataset
            self.dataset_path = '/media/nicolas/nicolas_seagate/datasets/kitti/stereo/stereo2015/data_scene_flow/'
        else:
            if test_file_path == '':
                file = 'data/' + self.name + '_' + mode + '.txt'
            else:
                file = test_file_path

        return file
