# =========== #
#  Libraries  #
# =========== #
import logging
import os


# ========= #
#  Classes  #
# ========= #
class Settings:
    def __init__(self, output_dir, output_tmp_dir, output_log_file):
        self.output_dir = output_dir
        self.output_tmp_dir = output_tmp_dir
        self.output_tmp_pred_dir = output_tmp_dir + 'pred/'
        self.output_tmp_gt_dir = output_tmp_dir + 'gt/'
        self.logger_output_file = output_dir + output_log_file

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for root, dirs, files in os.walk(self.output_tmp_dir, topdown=False):
            for name in files:
                # print(os.path.join(root, name))
                os.remove(os.path.join(root, name))
            for name in dirs:
                # print(os.path.join(root, name))
                os.rmdir(os.path.join(root, name))

        if not os.path.exists(self.output_tmp_pred_dir):
            os.makedirs(self.output_tmp_pred_dir)

        if not os.path.exists(self.output_tmp_gt_dir):
            os.makedirs(self.output_tmp_gt_dir)


# ================== #
#  Global Variables  #
# ================== #
settings = Settings('output/', 'output/tmp/', 'log.txt')

# Log config
logging.basicConfig(filename=settings.logger_output_file,
                    level=logging.DEBUG,
                    format='%(levelname)s %(asctime)s %(name)s %(funcName)s > %(message)s')

logger = logging.getLogger('log')
