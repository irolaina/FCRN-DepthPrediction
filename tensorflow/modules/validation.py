# ===========
#  Libraries
# ===========
import tensorflow as tf

from .dataloader import Dataloader
from modules.third_party.laina.fcrn import ResNet50UpProj
from .plot import Plot

# ==================
#  Global Variables
# ==================


# ===================
#  Class Declaration
# ===================
class Validation:
    def __init__(self, args, input_size, output_size, max_depth, dataset_name):
        # Raw Input/Output
        self.tf_image_key = tf.placeholder(tf.string)
        self.tf_depth_key = tf.placeholder(tf.string)

        tf_image_file = tf.read_file(self.tf_image_key)
        tf_depth_file = tf.read_file(self.tf_depth_key)

        if dataset_name == 'apolloscape':
            self.tf_image_raw = tf.image.decode_jpeg(tf_image_file, channels=3)
        else:
            self.tf_image_raw = tf.image.decode_png(tf_image_file, channels=3, dtype=tf.uint8)

        if dataset_name.split('_')[0] == 'kittidiscrete' or \
           dataset_name.split('_')[0] == 'kitticontinuous':
            self.tf_depth_raw = tf.image.decode_png(tf_depth_file, channels=1, dtype=tf.uint8)
        else:
            self.tf_depth_raw = tf.image.decode_png(tf_depth_file, channels=1, dtype=tf.uint16)

        # True Depth Value Calculation. May vary from dataset to dataset.
        tf_depth = Dataloader.rawdepth2meters(self.tf_depth_raw, args.dataset)

        # Network Input/Output. Overwrite Tensors!
        # tf_image = tf.cast(self.tf_image_raw, tf.float32)   # uint8 -> float32 [0.0, 255.0]
        tf_image = tf.image.convert_image_dtype(self.tf_image_raw, tf.float32)   # uint8 -> float32 [0.0, 1.0]
        self.tf_image = tf_image
        self.tf_depth = tf_depth

        # Crops Input and Depth Images (Removes Sky)
        if args.remove_sky:
            self.tf_image, self.tf_depth = Dataloader.removeSky(tf_image, tf_depth, dataset_name)

        # Downsizes Input and Depth Images
        self.tf_image_resized = tf.image.resize_images(self.tf_image, [input_size.height, input_size.width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
        self.tf_depth_resized = tf.image.resize_images(self.tf_depth, [output_size.height, output_size.width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

        # self.tf_image_resized_uint8 = tf.cast(self.tf_image_resized, tf.uint8)  # Visual Purpose
        self.tf_image_resized_uint8 = tf.image.convert_image_dtype(self.tf_image_resized, tf.uint8)  # Visual Purpose

        self.fcrn = ResNet50UpProj({'data': tf.expand_dims(self.tf_image_resized, axis=0)}, batch=args.batch_size, keep_prob=1, is_training=False)
        self.tf_pred = self.fcrn.get_output()

        # Clips predictions above a certain distance in meters. Inspired from Monodepth's article.
        # if max_depth is not None:
        #     self.tf_pred = tf.clip_by_value(self.tf_pred, 0, tf.constant(max_depth))

        with tf.name_scope('Valid'):
            self.tf_loss = None
            self.loss = -1
            self.loss_hist = []

        if args.show_valid_progress:
            self.plot = Plot(args.mode, title='Validation Prediction')

        print("[Network/Validation] Validation Tensors created.")
        print(self.tf_image)
        print(self.tf_depth)
        print(self.tf_image_resized)
        print(self.tf_image_resized_uint8)
        print(self.tf_depth_resized)
        print()
        # input("valid")
