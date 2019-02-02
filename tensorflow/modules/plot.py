# ===========
#  Libraries
# ===========
import numpy as np
import matplotlib.pyplot as plt
import modules.loss as loss
# from mpl_toolkits.axes_grid1 import make_axes_locatable


# ==================
#  Global Variables
# ==================


# ===========
#  Functions
# ===========
def updateColorBar(cbar, img):
    vmin, vmax = np.min(img), np.max(img)
    cbar.set_clim(vmin, vmax)

    cbar_ticks = np.linspace(vmin, vmax, num=7, endpoint=True)
    cbar.set_ticks(cbar_ticks)

    cbar.draw_all()

    # Debug
    # print("vmin:", vmin, "\tvmax:", vmax)


# ===================
#  Class Declaration
# ===================
class Plot(object):
    def __init__(self, mode, title):
        self.fig, self.axes = None, None
        self.cax, self.cbar = [], []

        if mode == 'train':  # and Validation
            self.fig, self.axes = plt.subplots(7, 1, figsize=(10, 5))
            self.axes[0] = plt.subplot(221)
            self.axes[1] = plt.subplot(222)
            self.axes[2] = plt.subplot(223)
            self.axes[3] = plt.subplot(224)

            # Sets Titles
            self.axes[0].set_title("Image")
            self.axes[1].set_title("Label")
            self.axes[2].set_title("Pred")
            self.axes[3].set_title("MSE(Pred)")

        elif mode == 'test':
            self.fig = plt.figure()
            # self.fig.subplots_adjust(bottom=0.1, left=0.1, top=0.9, right=0.9)
            X = [(3, 5, (1, 2)),
                 (3, 5, (6, 7)),
                 (3, 5, 3),
                 (3, 5, 8),
                 (3, 5, 13),
                 (3, 5, (11, 12)),
                 (3, 5, 14),
                 (3, 5, 15)]
            self.axes = []
            for nrows, ncols, plot_number in X:
                self.axes.append(self.fig.add_subplot(nrows, ncols, plot_number))

            # Sets Titles
            self.axes[0].set_title("Image")
            self.axes[1].set_title("Depth")
            self.axes[2].set_title("Image Resized")
            self.axes[3].set_title("Depth Resized")
            self.axes[4].set_title("Pred")
            self.axes[5].set_title("up(Pred)")
            self.axes[6].set_title("Pred_50")
            self.axes[7].set_title("Pred_80")

        self.fig.canvas.set_window_title(title)
        self.fig.tight_layout(pad=0.1, w_pad=None, h_pad=None)  # Fix Subplots Spacing

        # TODO: Terminar
        # Fix Colorbar size
        # self.divider0 = make_axes_locatable(self.axes[0])
        # self.divider1 = make_axes_locatable(self.axes[1])
        # self.divider2 = make_axes_locatable(self.axes[2])
        # self.divider3 = make_axes_locatable(self.axes[3])
        # self.divider4 = make_axes_locatable(self.axes[4])
        # self.divider5 = make_axes_locatable(self.axes[5])
        # self.divider6 = make_axes_locatable(self.axes[6])
        # self.divider7 = make_axes_locatable(self.axes[7])
        #
        # # self.cax0_div = self.divider0.append_axes("right", size="5%", pad=0.15)
        # self.cax1_div = self.divider1.append_axes("right", size="5%", pad=0.15)
        # # self.cax2_div = self.divider2.append_axes("right", size="5%", pad=0.15)
        # self.cax3_div = self.divider3.append_axes("right", size="5%", pad=0.15)
        # self.cax4_div = self.divider4.append_axes("right", size="5%", pad=0.15)
        # self.cax5_div = self.divider5.append_axes("right", size="5%", pad=0.15)
        # self.cax6_div = self.divider6.append_axes("right", size="5%", pad=0.15)
        # self.cax7_div = self.divider7.append_axes("right", size="5%", pad=0.15)

        self.is_first_time = True

    def show_train_results(self, raw, label, pred):
        pred_mse = loss.np_mse(y=pred, y_=label)

        if self.is_first_time:
            self.cax.append(self.axes[0].imshow(raw))
            self.cax.append(self.axes[1].imshow(label))
            self.cax.append(self.axes[2].imshow(pred))
            self.cax.append(self.axes[3].imshow(pred_mse, cmap='jet'))

            # Creates ColorBars
            self.cbar.append(None)
            self.cbar.append(self.fig.colorbar(self.cax[1], ax=self.axes[1]))
            self.cbar.append(self.fig.colorbar(self.cax[2], ax=self.axes[2]))
            self.cbar.append(self.fig.colorbar(self.cax[3], ax=self.axes[3]))

            self.is_first_time = False
        else:
            # Updates Colorbars
            updateColorBar(self.cbar[1], label)
            updateColorBar(self.cbar[2], pred)
            updateColorBar(self.cbar[3], pred_mse)

            # Updates Images
            self.cax[0].set_data(raw)
            self.cax[1].set_data(label)
            self.cax[2].set_data(pred)
            self.cax[3].set_data(pred_mse)
            plt.draw()

        plt.pause(0.001)

    def show_test_results(self, image, depth, image_resized, depth_resized, pred, pred_up, pred_50, pred_80, i):
        if self.is_first_time:
            self.cax.append(self.axes[0].imshow(image))
            self.cax.append(self.axes[1].imshow(depth))
            self.cax.append(self.axes[2].imshow(image_resized))
            self.cax.append(self.axes[3].imshow(depth_resized))
            self.cax.append(self.axes[4].imshow(pred))
            self.cax.append(self.axes[5].imshow(pred_up))
            self.cax.append(self.axes[6].imshow(pred_50))
            self.cax.append(self.axes[7].imshow(pred_80))

            # Creates ColorBars
            self.cbar.append(None)
            self.cbar.append(self.fig.colorbar(self.cax[1], ax=self.axes[1]))
            self.cbar.append(None)
            self.cbar.append(self.fig.colorbar(self.cax[3], ax=self.axes[3]))
            self.cbar.append(self.fig.colorbar(self.cax[4], ax=self.axes[4]))
            self.cbar.append(self.fig.colorbar(self.cax[5], ax=self.axes[5]))
            self.cbar.append(self.fig.colorbar(self.cax[6], ax=self.axes[6]))
            self.cbar.append(self.fig.colorbar(self.cax[7], ax=self.axes[7]))

            self.is_first_time = False
        else:
            # Updates Colorbars
            updateColorBar(self.cbar[1], depth)
            updateColorBar(self.cbar[3], depth_resized)
            updateColorBar(self.cbar[4], pred)
            updateColorBar(self.cbar[5], pred_up)
            updateColorBar(self.cbar[6], pred_50)
            updateColorBar(self.cbar[7], pred_80)

            # Updates Images
            self.cax[0].set_data(image)
            self.cax[1].set_data(depth)
            self.cax[2].set_data(image_resized)
            self.cax[3].set_data(depth_resized)
            self.cax[4].set_data(pred)
            self.cax[5].set_data(pred_up)
            self.cax[6].set_data(pred_50)
            self.cax[7].set_data(pred_80)

            plt.draw()

        self.fig.canvas.set_window_title("Test Predictions [%d]" % i)

        plt.pause(1)
