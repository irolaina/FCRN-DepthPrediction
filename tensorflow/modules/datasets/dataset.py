# ===========
#  Libraries
# ===========
from ..size import Size


# ===================
#  Class Declaration
# ===================
class Dataset(object):
    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop('name')

        dataset_root = kwargs.pop('dataset_root')
        height = kwargs.pop('height')
        width = kwargs.pop('width')
        self.max_depth = kwargs.pop('max_depth')  # Max Depth to limit predictions

        super(Dataset, self).__init__(*args, **kwargs)

        self.dataset_path = dataset_root + "kitti/raw_data/"

        self.image_size = Size(height, width, 3)
        self.depth_size = Size(height, width, 1)
