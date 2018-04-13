# ===========
#  Libraries
# ===========
from .size import Size

# ===================
#  Class Declaration
# ===================
class NyuDepth(object):
    def __init__(self, name, dataset_path):
        self.path = dataset_path
        self.name = name
        self.type = 'nyudepth'

        self.image_size = Size(480, 640, 3)
        self.depth_size = Size(480, 640, 1)

        print("[monodeep/Dataloader] NyuDepth object created.")

    # TODO: Terminar
    def getFileList(self):
        pass
