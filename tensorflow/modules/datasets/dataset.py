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
        self.dataset_path = kwargs.pop('dataset_path')
        self.name = kwargs.pop('name')
        height = kwargs.pop('height')
        width = kwargs.pop('width')
        self.max_depth = kwargs.pop('max_depth')  # Max Depth to limit predictions

        super(Dataset, self).__init__(**kwargs)

        self.image_size = Size(height, width, 3)
        self.depth_size = Size(height, width, 1)

        # Train/Test Split Ratio
        self.ratio = 0.8

        print("[Dataloader] %s object created." % self.name)

    # FIXME: Esta função está correta?
    # Acredito que ainda seja necessário arrumar a combinação dos eval_tool e os test_splits. Ou indepente qual eval tool está sendo usado quando não se especifica o test_split?
    def get_file_path(self, mode, test_split, test_file_path):
        # KITTI Stereo 2015: 200 Test Images
        # Eigen Split: 697 Test Images
        # Eigen Split & KITTI Depth: 652 Test Images

        # ------------------------------------------------------------- #
        #  Evaluation based on Disparity Images (Eval Tool: MonoDepth)  #
        # ------------------------------------------------------------- #
        if args.eval_tool == 'monodepth' and (test_split == 'eigen' or test_split == 'eigen_continuous'):
            file_path = 'modules/third_party/monodepth/utils/filenames/eigen_test_files.txt'

            # Overwrite the 'dataset_path' specified by the dataset
            self.dataset_path = '/media/nicolas/nicolas_seagate/datasets/kitti/raw_data/'

        elif args.eval_tool == 'monodepth' and test_split == 'kitti_stereo':
            file_path = 'modules/third_party/monodepth/utils/filenames/kitti_stereo_2015_test_files.txt'

            # Overwrite the 'dataset_path' specified by the dataset
            self.dataset_path = '/media/nicolas/nicolas_seagate/datasets/kitti/stereo/stereo2015/data_scene_flow/'

        elif args.eval_tool == 'monodepth' and test_split == 'eigen_kitti_depth':
            file_path = 'data/new_splits/eigen_split_based_on_kitti_depth/eigen_test_kitti_depth_files.txt'

            # Overwrite the 'dataset_path' specified by the dataset
            self.dataset_path = '/media/nicolas/nicolas_seagate/datasets/kitti/'

        # --------------------------------------------------------------------------------- #
        #  Evaluation based on Ground Truth/Velodyne Scans Images (Eval Tool: KITTI Depth)  #
        # --------------------------------------------------------------------------------- #
        # FIXME:
        elif args.eval_tool == 'kitti_depth' and (test_split == 'eigen' or test_split == 'eigen_continuous'):
            print("Não deveria rodar! Terminar Implementação. Devo gerar os mapas de profundidade para que possa ser avaliado.")
            raise SystemError

        # FIXME:
        elif args.eval_tool == 'kitti_depth' and test_split == 'kitti_stereo':
            print("Não deveria rodar! Terminar Implementação. Devo gerar os mapas de profundidade para que possa ser avaliado.")
            raise SystemError

        elif args.eval_tool == 'kitti_depth' and test_split == 'eigen_kitti_depth':
            file_path = 'data/new_splits/eigen_split_based_on_kitti_depth/eigen_test_kitti_depth_files.txt'

            # Overwrite the 'dataset_path' specified by the dataset
            self.dataset_path = '/media/nicolas/nicolas_seagate/datasets/kitti/'

        else:
            if test_file_path == '':
                file_path = 'data/' + self.name + '_' + mode + '.txt'
            else:
                file_path = test_file_path

        args.test_file_path = file_path

        return file_path
