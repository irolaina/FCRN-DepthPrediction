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

        if mode == 'train': # Or Validation
            self.fig, self.axes = plt.subplots(5, 1)
            self.axes[0] = plt.subplot(231)
            self.axes[1] = plt.subplot(232)
            self.axes[2] = plt.subplot(233)
            self.axes[3] = plt.subplot(234)
            self.axes[4] = plt.subplot(235)

            # Sets Titles
            self.axes[0].set_title("Raw")
            self.axes[1].set_title("Label")
            self.axes[2].set_title("log(Label)")
            self.axes[3].set_title("Pred")
            self.axes[4].set_title("MSE(Pred)")

        elif mode == 'test':
            self.fig, self.axes = plt.subplots(4, 1)
            self.axes[0] = plt.subplot(221)
            self.axes[1] = plt.subplot(223)
            self.axes[2] = plt.subplot(222)
            self.axes[3] = plt.subplot(224)

            # Sets Titles
            self.axes[0].set_title("Raw")
            self.axes[1].set_title("Label")
            self.axes[2].set_title("log(Label)")
            self.axes[3].set_title("Pred")

        self.fig.canvas.set_window_title(title)
        self.fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)  # Fix Subplots Spacing

        self.isFirstTime = True

    # TODO: Add colorbar
    # TODO: Otimizar
    def showTrainResults(self, raw, label, log_label, pred):
        predMSE = loss.np_MSE(y=pred, y_=log_label)

        if self.isFirstTime:
            self.cax1 = self.axes[0].imshow(raw)
            self.cax2 = self.axes[1].imshow(label)
            self.cax3 = self.axes[2].imshow(log_label)
            self.cax4 = self.axes[3].imshow(pred)
            self.cax5 = self.axes[4].imshow(predMSE, cmap='jet')

            # cbar2 = self.fig.colorbar(self.cax2, ax=self.axes[1])
            # # self.axes[1].set_aspect('auto')
            #
            # cbar3 = self.fig.colorbar(self.cax3, ax=self.axes[2])
            # # self.axes[2].set_aspect('auto')
            #
            # cbar4 = self.fig.colorbar(self.cax4, ax=self.axes[3])
            # # self.axes[3].set_aspect('auto')
            #
            # cbar5 = self.fig.colorbar(self.cax5, ax=self.axes[4])
            # # self.axes[4].set_aspect('auto')

            self.isFirstTime = False
        else:
            # self.cax1.set_data(raw)
            # self.cax2.set_data(label)
            # self.cax3.set_data(log_label)
            # self.cax4.set_data(pred)
            # self.cax5.set_data(predMSE)
            # plt.draw()

            self.cax1 = self.axes[0].imshow(raw)
            self.cax2 = self.axes[1].imshow(label)
            self.cax3 = self.axes[2].imshow(log_label)
            self.cax4 = self.axes[3].imshow(pred)
            self.cax5 = self.axes[4].imshow(predMSE, cmap='jet')

        plt.pause(0.001)

    # TODO: Add colorbar
    def showValidResults(self, raw, label, log_label, pred):
        predMSE = loss.np_MSE(y=pred, y_=log_label)

        if self.isFirstTime:
            self.cax1 = self.axes[0].imshow(raw)
            self.cax2 = self.axes[1].imshow(label)
            self.cax3 = self.axes[2].imshow(log_label)
            self.cax4 = self.axes[3].imshow(pred)
            self.cax5 = self.axes[4].imshow(predMSE, cmap='jet')

            # cbar2 = self.fig.colorbar(self.cax2, ax=self.axes[1])
            # # self.axes[1].set_aspect('auto')
            #
            # cbar3 = self.fig.colorbar(self.cax3, ax=self.axes[2])
            # # self.axes[2].set_aspect('auto')
            #
            # cbar4 = self.fig.colorbar(self.cax4, ax=self.axes[3])
            # # self.axes[3].set_aspect('auto')
            #
            # cbar5 = self.fig.colorbar(self.cax5, ax=self.axes[4])
            # # self.axes[4].set_aspect('auto')

            self.isFirstTime = False

        else:
            # self.cax1.set_data(raw)
            # self.cax2.set_data(label)
            # self.cax3.set_data(log_label)
            # self.cax4.set_data(pred)
            # self.cax5.set_data(predMSE)
            # plt.draw()

            self.cax1 = self.axes[0].imshow(raw)
            self.cax2 = self.axes[1].imshow(label)
            self.cax3 = self.axes[2].imshow(log_label)
            self.cax4 = self.axes[3].imshow(pred)
            self.cax5 = self.axes[4].imshow(predMSE, cmap='jet')

        plt.pause(0.001)

    # TODO: Add colorbar
    def showTestResults(self, raw, label, log_label, pred, i):
        if self.isFirstTime:
            self.cax1 = self.axes[0].imshow(raw)
            self.cax2 = self.axes[1].imshow(label)
            self.cax3 = self.axes[2].imshow(log_label)
            self.cax4 = self.axes[3].imshow(pred)

            # self.fig.colorbar(self.cax2, ax=self.axes[1])
            # self.fig.colorbar(self.cax3, ax=self.axes[2])
            # self.fig.colorbar(self.cax4, ax=self.axes[3])

            self.isFirstTime = False
        else:
            self.cax1.set_data(raw)
            self.cax2.set_data(label)
            self.cax3.set_data(log_label)
            self.cax4.set_data(pred)
            plt.draw()

        self.fig.canvas.set_window_title("Test Predictions [%d]" % i)

        plt.pause(0.5)

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
