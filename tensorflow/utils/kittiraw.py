# ===========
#  Libraries
# ===========
from .size import Size

# ===================
#  Class Declaration
# ===================
class KittiRaw(object):
    def __init__(self, name, dataset_path):
        self.path = dataset_path
        self.name = name
        self.type = 'kittiraw'

        # Informações abaixo são do kittiraw_residential_continuous
        self.image_size = Size(375, 1242, 3)
        self.depth_size = Size(375, 1242, 1)

        print("[fcrn/Dataloader] Kitti object created.")

    # TODO: Terminar
    def getFileList(self):
        pass