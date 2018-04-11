# ===================
#  Class Declaration
# ===================
class NyuDepth(object):
    def __init__(self, name, dataset_path):
        self.path = dataset_path
        self.name = name
        self.type = 'nyudepth'

        self.imageInputSize = [480, 640]
        self.depthInputSize = [480, 640]

        # Monodeep
        # self.imageNetworkInputSize = [228, 304]
        # self.depthNetworkOutputSize = [57, 76]
        # self.depthBilinearOutputSize = [228, 304]

        # FCRN
        self.imageNetworkInputSize = [228, 304]
        self.depthNetworkOutputSize = [128, 160]

        print("[monodeep/Dataloader] NyuDepth object created.")

    # TODO: Terminar
    def getFileList(self):
        pass
