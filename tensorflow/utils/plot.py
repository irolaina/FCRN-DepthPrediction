# ===========
#  Libraries
# ===========
import matplotlib.pyplot as plt
import utils.loss as loss


# ==================
#  Global Variables
# ==================

# ===================
#  Class Declaration
# ===================
class Plot(object):
    def __init__(self, mode, title):
        self.fig, self.axes = None, None

        if mode == 'train':
            self.fig, self.axes = plt.subplots(5, 1)
            self.axes[0] = plt.subplot(231)
            self.axes[1] = plt.subplot(232)
            self.axes[2] = plt.subplot(233)
            self.axes[3] = plt.subplot(234)
            self.axes[4] = plt.subplot(235)

        elif mode == 'test':
            self.fig, self.axes = plt.subplots(4, 1)
            self.axes[0] = plt.subplot(221)
            self.axes[1] = plt.subplot(223)
            self.axes[2] = plt.subplot(222)
            self.axes[3] = plt.subplot(224)

        self.fig.canvas.set_window_title(title)
        self.isFirstTime = True

    # TODO: Add colorbar
    def showTrainResults(self, raw, label, log_label, pred):
        plt.figure(1)

        # Set Titles and subplots spacing. Runs only at first Time
        if self.isFirstTime:
            self.axes[0].set_title("Raw")
            self.axes[1].set_title("Label")
            self.axes[2].set_title("log(Label)")
            self.axes[3].set_title("Pred")
            self.axes[4].set_title("MSE(Pred)")
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

            self.isFirstTime = False

        self.axes[0].imshow(raw)

        cax1 = self.axes[1].imshow(label)
        # self.fig.colorbar(cax1, ax=self.axes[1])

        cax2 = self.axes[2].imshow(log_label)
        # self.fig.colorbar(cax2, ax=self.axes[2])

        cax3 = self.axes[3].imshow(pred)
        # self.fig.colorbar(cax3, ax=self.axes[3])

        # TODO: Lembre que a Training Loss utilizaRMSE_log_scaleInv, porém o resultado é avaliado utilizando MSE
        predMSE = loss.np_MSE(y=pred, y_=log_label)
        cax4 = self.axes[4].imshow(predMSE, cmap='jet')
        # self.fig.colorbar(cax4)

        plt.pause(0.001)

    # TODO: Add colorbar
    def showValidResults(self, raw, label, log_label, pred):
        plt.figure(2)

        # Set Titles and subplots spacing. Runs only at first Time
        if self.isFirstTime:
            self.axes[0].set_title("Raw")
            self.axes[1].set_title("Label")
            self.axes[2].set_title("log(Label)")
            self.axes[3].set_title("Pred")
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

            self.isFirstTime = False

        self.axes[0].imshow(raw)
        cax1 = self.axes[1].imshow(label)
        # self.fig.colorbar(cax1, ax=self.axes[1])

        cax2 = self.axes[2].imshow(log_label)
        # self.fig.colorbar(cax2, ax=self.axes[2])

        cax3 = self.axes[3].imshow(pred)
        # self.fig.colorbar(cax3, ax=self.axes[3])

        plt.pause(0.001)

    # TODO: Add colorbar
    def showTestResults(self, raw, label, log_label, pred, i):
        plt.figure(1)

        # Set Titles and subplots spacing. Runs only at first Time
        if self.isFirstTime:
            self.axes[0].set_title("Raw")
            self.axes[1].set_title("Label")
            self.axes[2].set_title("log(Label)")
            self.axes[3].set_title("Pred")
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            self.isFirstTime = False

        self.fig.canvas.set_window_title("Test Predictions [%d]" % i)

        self.axes[0].imshow(raw)
        cax1 = self.axes[1].imshow(label)
        # self.fig.colorbar(cax1, ax=self.axes[1])

        cax2 = self.axes[2].imshow(log_label)
        # self.fig.colorbar(cax2, ax=self.axes[2])

        cax3 = self.axes[3].imshow(pred)
        # self.fig.colorbar(cax3, ax=self.axes[3])

        plt.pause(0.001)

    # TODO: Remover Deprecated
    @staticmethod
    # TODO: Add raw, log_labels, coarse
    def showValidationProgress(label, fine, figId):
        fig = plt.figure(figId)
        fig.clf()
        ax = fig.add_subplot(2, 1, 1)
        cax = ax.imshow(label, cmap='gray')
        plt.title("valid_labels[0,:,:]")
        fig.colorbar(cax)

        ax = fig.add_subplot(2, 1, 2)
        cax = ax.imshow(fine, cmap='gray')
        plt.title("vPredictions_f[0,:,:]")
        fig.colorbar(cax)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)