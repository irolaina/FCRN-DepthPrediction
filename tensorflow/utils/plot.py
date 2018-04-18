# ===========
#  Libraries
# ===========
import numpy as np
import matplotlib.pyplot as plt
import utils.loss as loss


# ==================
#  Global Variables
# ==================


# ===========
#  Functions
# ===========
def updateColorBar(cbar, img):
    vmin, vmax = np.min(img), np.max(img)
    cbar.set_clim(vmin, vmax)
    # new_cbar_ticks = np.linspace(vmin, vmax, num=6, endpoint=True)
    # cbar.set_ticks(new_cbar_ticks)


# ===================
#  Class Declaration
# ===================
class Plot(object):
    def __init__(self, mode, title):
        self.fig, self.axes = None, None

        if mode == 'train':  # Or Validation
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
        # self.fig.set_size_inches(9, 5)
        self.fig.tight_layout(pad=0.4, w_pad=2, h_pad=1.0)  # Fix Subplots Spacing

        self.isFirstTime = True

    def showResults(self, raw, label, log_label, pred, cbar_range):
        predMSE = loss.np_MSE(y=pred, y_=log_label)

        if self.isFirstTime:
            self.cax1 = self.axes[0].imshow(raw)
            self.cax2 = self.axes[1].imshow(label, vmin=cbar_range.vmin, vmax=cbar_range.vmax)
            self.cax3 = self.axes[2].imshow(log_label, vmin=cbar_range.log_vmin, vmax=cbar_range.log_vmax)
            self.cax4 = self.axes[3].imshow(pred, vmin=cbar_range.log_vmin, vmax=cbar_range.log_vmax)
            self.cax5 = self.axes[4].imshow(predMSE, cmap='jet')

            # Creates ColorBars
            self.cbar2 = self.fig.colorbar(self.cax2, ax=self.axes[1])
            self.cbar3 = self.fig.colorbar(self.cax3, ax=self.axes[2])
            self.cbar4 = self.fig.colorbar(self.cax4, ax=self.axes[3])
            self.cbar5 = self.fig.colorbar(self.cax5, ax=self.axes[4])

            self.isFirstTime = False
        else:
            # Updates Colorbars
            updateColorBar(self.cbar2, label)
            updateColorBar(self.cbar3, log_label)
            updateColorBar(self.cbar4, pred)
            updateColorBar(self.cbar5, predMSE)

            # Updates Images
            self.cax1.set_data(raw)
            self.cax2.set_data(label)
            self.cax3.set_data(log_label)
            self.cax4.set_data(pred)
            self.cax5.set_data(predMSE)
            plt.draw()

        plt.pause(0.001)

    def showTestResults(self, raw, label, log_label, pred, i):
        # predMSE = loss.np_MSE(y=pred, y_=log_label)

        if self.isFirstTime:
            self.cax1 = self.axes[0].imshow(raw)
            self.cax2 = self.axes[1].imshow(label)
            self.cax3 = self.axes[2].imshow(log_label)
            self.cax4 = self.axes[3].imshow(pred)
            # self.cax5 = self.axes[4].imshow(predMSE, cmap='jet')


            # Creates ColorBars
            self.cbar2 = self.fig.colorbar(self.cax2, ax=self.axes[1])
            self.cbar3 = self.fig.colorbar(self.cax3, ax=self.axes[2])
            self.cbar4 = self.fig.colorbar(self.cax4, ax=self.axes[3])

            self.isFirstTime = False
        else:
            # Updates Colorbars
            updateColorBar(self.cbar2, label)
            updateColorBar(self.cbar3, log_label)
            updateColorBar(self.cbar4, pred)
            # updateColorBar(self.cbar5, predMSE)

            # Updates Images
            self.cax1.set_data(raw)
            self.cax2.set_data(label)
            self.cax3.set_data(log_label)
            self.cax4.set_data(pred)
            # self.cax5.set_data(predMSE)
            plt.draw()

        self.fig.canvas.set_window_title("Test Predictions [%d]" % i)

        plt.pause(0.5)
