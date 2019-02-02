# =========== #
#  Libraries  #
# =========== #
import logging
import os

import pandas as pd


# =========== #
#  Functions  #
# =========== #
class Settings:
    def __init__(self, output_dir, output_log_file):
        self.output_dir = output_dir
        self.logger_output_file = output_dir + output_log_file

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


# Log config
settings = Settings('output/', 'log.txt')
logging.basicConfig(filename=settings.logger_output_file,
                    level=logging.DEBUG,
                    format='%(levelname)s %(asctime)s %(name)s %(funcName)s > %(message)s')

logger = logging.getLogger('log')