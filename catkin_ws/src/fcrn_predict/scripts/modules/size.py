# ===========
#  Libraries
# ===========


# ===================
#  Class Declaration
# ===================
class Size:
    def __init__(self, height, width, nchannels):
        self.height = height
        self.width = width
        self.nchannels = nchannels

        # print("Size object created.")

    def getSize(self):
        return self.height, self.width, self.nchannels
