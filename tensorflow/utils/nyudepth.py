# ===========
#  Libraries
# ===========
from .size import Size


# ===================
#  Class Declaration
# ===================
class NyuDepth(object):
    def __init__(self):
        self.dataset_path = ''  # TODO: Terminar
        self.name = 'nyudepth'

        self.image_size = Size(480, 640, 3)
        self.depth_size = Size(480, 640, 1)

        print("[Dataloader] NyuDepth object created.")

    # TODO: Terminar
    def getFileList(self):
        pass
