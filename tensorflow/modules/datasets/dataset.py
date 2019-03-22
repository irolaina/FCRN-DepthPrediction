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
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()

        self.dataset_path = self.get_dataset_root() + kwargs.pop('dataset_rel_path')
        self.name = kwargs.pop('name')
        height = kwargs.pop('height')
        width = kwargs.pop('width')
        self.max_depth = kwargs.pop('max_depth')  # Max Depth to limit predictions
        self.image_size = Size(height, width, 3)
        self.depth_size = Size(height, width, 1)
        self.ratio = 0.8  # Train/Test Split Ratio

        print("[Dataloader] %s object created." % self.name)

    @staticmethod
    def get_dataset_root():
        """ Defines dataset_root path depending on which machine is used."""
        dataset_root = None

        if args.machine == 'nicolas':
            dataset_root = "/media/nicolas/nicolas_seagate/datasets/"
        elif args.machine == 'olorin':
            dataset_root = "/media/olorin/Documentos/datasets/"

        return dataset_root

    def get_file_path(self, mode, test_split, test_file_path):
        # KITTI Stereo 2015: 200 Test Images
        # Eigen Split: 697 Test Images
        # Eigen Split & KITTI Depth: 652 Test Images
        file_path = None

        # ------------------------------------------------------------- #
        #  Evaluation based on Disparity Images (Eval Tool: MonoDepth)  #
        # ------------------------------------------------------------- #
        if (args.mode == 'train' or args.mode == 'test') and test_split == '':  # Default
            file_path = 'data/' + self.name + '_' + mode + '.txt' if test_file_path == '' else test_file_path

        elif args.mode == 'test' and args.eval_tool == 'monodepth':
            if test_split == 'kitti_stereo':
                file_path = 'data/kitti_stereo_2015_test_files.txt'

                # Overwrite the 'dataset_path' specified by the dataset
                self.dataset_path = self.get_dataset_root() + 'kitti/stereo/stereo2015/data_scene_flow/'

            elif test_split == 'eigen':
                file_path = 'data/eigen_test_files.txt'

                # Overwrite the 'dataset_path' specified by the dataset
                self.dataset_path = self.get_dataset_root() + 'kitti/raw_data/'

            elif test_split == 'eigen_kitti_depth':
                file_path = 'data/eigen_test_kitti_depth_files.txt'

                # Overwrite the 'dataset_path' specified by the dataset
                self.dataset_path = self.get_dataset_root() + 'kitti/'
            else:
                raise ValueError('')

        # --------------------------------------------------------------------------------- #
        #  Evaluation based on Ground Truth/Velodyne Scans Images (Eval Tool: KITTI Depth)  #
        # --------------------------------------------------------------------------------- #
        elif args.mode == 'test' and args.eval_tool == 'kitti_depth':
            if test_split == 'kitti_stereo' or test_split == 'eigen':  # FIXME:
                raise NotImplementedError("Não deveria rodar! Terminar Implementação. Devo gerar os mapas de profundidade para que possa ser avaliado.")

            elif test_split == 'eigen_kitti_depth':
                file_path = 'data/new_splits/eigen_split_based_on_kitti_depth/eigen_test_kitti_depth_files.txt'

                # Overwrite the 'dataset_path' specified by the dataset
                self.dataset_path = self.get_dataset_root() + 'kitti/'

        args.test_file_path = file_path

        return file_path
