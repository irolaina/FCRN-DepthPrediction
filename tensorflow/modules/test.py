# ===========
#  Libraries
# ===========
import tensorflow as tf

from .third_party.laina.fcrn import ResNet50UpProj
from .size import Size

# ==================
#  Global Variables
# ==================


# ===================
#  Class Declaration
# ===================
class Test:
    def __init__(self, args, data):
        # Construct the network
        with tf.variable_scope('model'):
            input_size = Size(228, 304, 3)
            output_size = Size(128, 160, 1)
            batch_size = 1

            self.tf_image_key = tf.placeholder(tf.string)
            self.tf_depth_key = tf.placeholder(tf.string)

            tf_image_file = tf.read_file(self.tf_image_key)
            tf_depth_file = tf.read_file(self.tf_depth_key)

            if data.dataset_name == 'apolloscape':
                tf_image = tf.image.decode_jpeg(tf_image_file, channels=3)
            else:
                tf_image = tf.image.decode_png(tf_image_file, channels=3, dtype=tf.uint8)

            if data.dataset_name.split('_')[0] == 'kittidiscrete' or \
                    data.dataset_name.split('_')[0] == 'kitticontinuous':
                tf_depth = tf.image.decode_png(tf_depth_file, channels=1, dtype=tf.uint8)
            else:
                tf_depth = tf.image.decode_png(tf_depth_file, channels=1, dtype=tf.uint16)

            if args.remove_sky:
                # Crops Input and Depth Images (Removes Sky)
                if data.dataset_name[0:5] == 'kitti':
                    tf_image_shape = tf.shape(tf_image)
                    tf_depth_shape = tf.shape(tf_depth)

                    crop_height_perc = tf.constant(0.3, tf.float32)
                    tf_image_new_height = crop_height_perc * tf.cast(tf_image_shape[0], tf.float32)
                    tf_depth_new_height = crop_height_perc * tf.cast(tf_depth_shape[0], tf.float32)

                    tf_image = tf_image[tf.cast(tf_image_new_height, tf.int32):, :]
                    tf_depth = tf_depth[tf.cast(tf_depth_new_height, tf.int32):, :]

            # True Depth Value Calculation. May vary from dataset to dataset.
            tf_depth = data.rawdepth2meters(tf_depth, data.dataset_name)

            # tf_image.set_shape(input_size.getSize())
            # tf_depth.set_shape(output_size.getSize())

            # Downsizes Input and Depth Images
            tf_image_resized = tf.image.resize_images(tf_image, [input_size.height, input_size.width])
            tf_image_resized_uint8 = tf.cast(tf_image_resized, tf.uint8)  # Visual purpose
            tf_image_resized = tf.expand_dims(tf_image_resized, axis=0)  # Model's Input size requirement

            tf_depth_resized = tf.image.resize_images(tf_depth, [output_size.height, output_size.width])

            net = ResNet50UpProj({'data': tf_image_resized}, batch=batch_size, keep_prob=1, is_training=False)
            tf_pred = net.get_output()

            tf_pred_up = tf.image.resize_images(tf_pred, tf.shape(tf_depth)[:2], tf.image.ResizeMethod.BILINEAR, False)

            # Group Tensors
            self.image_op = [self.tf_image_key, tf_image, tf_image_resized_uint8]
            self.depth_op = [self.tf_depth_key, tf_depth, tf_depth_resized]
            self.pred_op = [tf_pred, tf_pred_up]

            print("\n[Network/Test] Testing Tensors created.")
            print("\nTensors:")
            print(self.tf_image_key)
            print(self.tf_depth_key)
            print(tf_image)
            print(tf_depth)
            print(tf_image_resized)
            print(tf_image_resized_uint8)
            print(tf_pred)
            print(tf_pred_up)
