# ===========
#  Libraries
# ===========
from .size import Size


# ===================
#  Class Declaration
# ===================
class Kitti(object):
    def __init__(self):
        self.dataset_path = ''  # TODO: Terminar
        self.name = 'kitti'

        self.image_size = Size(376, 1241, 3)
        self.depth_size = Size(376, 1226, 1)

        print("[Dataloader] Kitti object created.")

    # TODO: Terminar
    def getFileList(self):
        pass
