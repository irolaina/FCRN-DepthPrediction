import argparse
import numpy as np
import tensorflow as tf
import os.path as osp
from matplotlib import pyplot as plt
from PIL import Image

import models

def predict(model_data_path, image_path):
   
    print('Loading the model')

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, 228, 304, 3))
    batch_size = 1
    
    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')
        net.load(model_data_path, sess)      
        
        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:               
                uninitialized_vars.append(var)

        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        sess.run(init_new_vars_op)
        
        # Read image
        img = Image.open(image_path)
        img.show
        img = np.array(img).astype('float32')
        img = np.expand_dims(np.asarray(img), axis = 0)
        
        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        
        # Plot result
        fig = plt.figure()
        ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        fig.colorbar(ii)
        plt.show()

        return pred
        
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    predict(args.model_path, args.image_paths)

if __name__ == '__main__':
    main()

        



