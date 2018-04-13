# ===========
#  Libraries
# ===========
from .size import Size

# ===================
#  Class Declaration
# ===================
class KittiRaw(object):
    def __init__(self):
        self.dataset_path = '' # TODO: Terminar
        self.name = 'kittiraw'

        self.image_size = Size(375, 1242, 3)
        self.depth_size = Size(375, 1242, 1)

        print("[fcrn/Dataloader] KittiRaw object created.")

    # TODO: Terminar
    def getFileList(self):
        pass