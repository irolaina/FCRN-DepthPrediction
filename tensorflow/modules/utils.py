# ===========
#  Libraries
# ===========
import glob
import os
import time
from collections import deque
from itertools import chain
from sys import getsizeof, stderr

from modules.args import args


# ========= #
#  Classes  #
# ========= #
class Settings:
    def __init__(self, app_name, output_dir, output_tmp_dir, output_log_file):
        self.app_name = app_name
        self.datetime = time.strftime("%Y-%m-%d") + '_' + time.strftime("%H-%M-%S")

        # Defines folders paths for saving the model variables to disk.
        px_str = args.px + '_px'
        relative_save_path = output_dir + self.app_name + '/' + args.dataset + '/' + px_str + '/' + args.loss + '/' + self.datetime + '/'
        self.save_path = os.path.join(os.getcwd(), relative_save_path)
        self.save_restore_path = os.path.join(self.save_path, 'restore/')

        self.output_dir = output_dir
        self.output_tmp_dir = output_tmp_dir
        self.output_tmp_pred_dir = output_tmp_dir + 'pred/'
        self.output_tmp_gt_dir = output_tmp_dir + 'gt/'
        self.log_tb = self.save_path + args.log_directory

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

        if not os.path.exists(self.save_restore_path):
            os.makedirs(self.save_restore_path)

    def get_save_path(self):
        return self.save_path

    def get_save_restore_path(self):
        return self.save_restore_path


# ===========
#  Functions
# ===========
def detect_available_models():
    if args.model_path == '':
        found_models = glob.glob(settings.output_dir + "fcrn/*/*/*/*/restore/*.meta")
        found_models.sort()

        for i, model in enumerate(found_models):
            print(i, model)

        selected_model_id = input("\nSelect Model: ")
        print()
        selected_model_path = os.path.splitext(found_models[int(selected_model_id)])[0]
    else:
        selected_model_path = args.model_path

    return selected_model_path


def total_size(o, handlers=None, verbose=False):
    """Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """

    if handlers is None:
        handlers = {}

    def dict_handler(d):
        return chain.from_iterable(d.items())

    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(var):
        if id(var) in seen:  # do not double count the same object
            return 0
        seen.add(id(var))
        s = getsizeof(var, default_size)

        if verbose:
            print(s, type(var), repr(var), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(var, typ):
                # noinspection PyCallingNonCallable
                s += sum(map(sizeof, handler(var)))
                break
        return s

    return sizeof(o)


# ================== #
#  Global Variables  #
# ================== #
settings = Settings('fcrn', 'output/', 'output/tmp/', 'log.txt')
