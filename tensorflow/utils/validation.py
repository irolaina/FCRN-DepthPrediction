# ===========
#  Libraries
# ===========
import tensorflow as tf

# ==================
#  Global Variables
# ==================
LOSS_LOG_INITIAL_VALUE = 0.1


# ===================
#  Class Declaration
# ===================
class Validation:
    def __init__(self, inputSize, outputSize):
        # Raw Input/Output
        self.tf_image = tf.placeholder(tf.float32,
                                       shape=(None, 375, 1242, 3))  # TODO: Usar variáveis com essas informações
        self.tf_depth = tf.placeholder(tf.float32,
                                       shape=(None, 375, 1242, 1))  # TODO: Usar variáveis com essas informações

        # Network Input/Output
        self.tf_image_resized = tf.image.resize_images(self.tf_image, [inputSize.height, inputSize.width])
        self.tf_depth_resized = tf.image.resize_images(self.tf_depth, [outputSize.height, outputSize.width])
        self.tf_log_depth_resized = tf.log(self.tf_depth_resized + tf.constant(LOSS_LOG_INITIAL_VALUE, dtype=tf.float32))

        self.loss = -1

        print("\n[Network/Validation] Validation Tensors Created.")
        print(self.tf_image)
        print(self.tf_depth)
        print(self.tf_image_resized)
        print(self.tf_depth_resized)
        print(self.tf_log_depth_resized)
