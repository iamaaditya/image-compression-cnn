# Multi-structure Regions of Interest
# 
# References : 
#       CNN structure based on VGG16, https://github.com/ry/tensorflow-vgg16/blob/master/vgg16.py
#       Channel independent feature maps (3D features) using https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#depthwise_conv2d_native 
#       GAP based on https://github.com/jazzsaxmafia/Weakly_detector/blob/master/src/detector.py
#       Conv2d layer based on https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py

import tensorflow as tf
import numpy as np
import cPickle
from params import CNNParams, HyperParams

hyper     = HyperParams(verbose=False)
cnn_param = CNNParams(verbose=False)

def print_model_params(verbose=True):
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        if verbose: print("name: " + str(variable.name) + " - shape:" + str(shape))
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        if verbose: print("variable parameters: " , variable_parametes)
        total_parameters += variable_parametes
    if verbose: print("total params: ", total_parameters)
    return total_parameters

class CNN():

    def load_vgg_weights(self):
        with open(hyper.vgg_weights) as f:
            self.pretrained_weights = cPickle.load(f)

    def get_vgg_weights(self, layer_name, bias=False):
        layer = self.pretrained_weights[layer_name]
        if bias: return layer[1]
        # tranpose because VGG weights were stored in diffeerent order
        return layer[0].transpose((2,3,1,0)) 

    def conv2d_depth_or_not(self, input_, name, nonlinearity=None):
        with tf.variable_scope(name) as scope:
            
            W_shape = cnn_param.layer_shapes[name + '/W']
            b_shape = cnn_param.layer_shapes[name + '/b']
            
            if hyper.fine_tuning and name not in ['conv6', 'conv6_1', 'depth']:
                # because conv6, conv6_1, and depth are the layers added on top of VGG 
                # hence not present in VGG 
                W = self.get_vgg_weights(name)
                b = self.get_vgg_weights(name, bias=True)
                W_initializer = tf.constant_initializer(W)
                b_initializer = tf.constant_initializer(b)
            else:
                W_initializer = tf.truncated_normal_initializer(stddev=hyper.stddev)
                b_initializer = tf.constant_initializer(0.0)
                
            conv_weights = tf.get_variable("W", shape=W_shape, initializer=W_initializer)
            conv_biases  = tf.get_variable("b", shape=b_shape, initializer=b_initializer)

            if name == 'depth':
                # learn different filter for each input channel
                # thus the number of input channel has to be reduced
                conv = tf.nn.depthwise_conv2d_native(input_, conv_weights, [1,1,1,1], padding='SAME')
                # conv = tf.nn.separable_conv2d(input_, conv_weights, [1,1,1,1], padding='SAME')
            else:
                conv = tf.nn.conv2d(input_, conv_weights, [1,1,1,1], padding='SAME')

            bias = tf.nn.bias_add(conv, conv_biases)
            bias = tf.nn.dropout(bias,0.7) 
            if nonlinearity is None: 
                return bias
            return nonlinearity(bias, name=name)

    # currently not required, but for experimentation purposes
    # there are two FCL layer at the end of VGG NET
    def fully_connected_layer(self, input_, input_size, output_size, name, nonlinearity=None):
        shape = input_.get_shape().to_list()
        x = tf.reshape(input_, [-1, np.prod(shape[1:])])
        with tf.variable_scope(name) as scope:
            W   = tf.get_variable("W", shape=[input_size, output_size], 
                  initializer=tf.random_normal_initializer(stddev=hyper.stddev))
            b   = tf.get_variable("b", shape=[output_size], initializer=tf.constant_initializer(0.))
            bias = tf.nn.bias_add(tf.matmul(x, W), b, name=scope)
            if nonlinearity is None: 
                return bias
            return nonlinearity(bias, name=name)
        return nonlinearity(bias, name=name)
       
    def image_conversion_scaling(self, image):
        # Conversion to bgr and mean substraction is common with VGGNET
        # Because pre-trained values use them, https://arxiv.org/pdf/1409.1556.pdf
        image *= 255.
        r, g, b = tf.split(image, 3, 3)
        VGG_MEAN = [103.939, 116.779, 123.68]
        return tf.concat([b-VGG_MEAN[0], g-VGG_MEAN[1], r-VGG_MEAN[2]], 3)


    def build(self, image):

        image = self.image_conversion_scaling(image)

        conv1_1    = self.conv2d_depth_or_not(image,   "conv1_1", nonlinearity=tf.nn.relu)
        conv1_2    = self.conv2d_depth_or_not(conv1_1, "conv1_2", nonlinearity=tf.nn.relu)
        pool1      = tf.nn.max_pool(conv1_2, ksize=cnn_param.pool_window, 
                     strides=cnn_param.pool_stride, padding='SAME', name='pool1')

        conv2_1    = self.conv2d_depth_or_not(pool1,   "conv2_1", nonlinearity=tf.nn.relu)
        conv2_2    = self.conv2d_depth_or_not(conv2_1, "conv2_2", nonlinearity=tf.nn.relu)
        pool2      = tf.nn.max_pool(conv2_2, ksize=cnn_param.pool_window,
                     strides=cnn_param.pool_stride, padding='SAME', name='pool2')

        conv3_1    = self.conv2d_depth_or_not(pool2,   "conv3_1", nonlinearity=tf.nn.relu)
        conv3_2    = self.conv2d_depth_or_not(conv3_1, "conv3_2", nonlinearity=tf.nn.relu)
        conv3_3    = self.conv2d_depth_or_not(conv3_2, "conv3_3", nonlinearity=tf.nn.relu)
        pool3      = tf.nn.max_pool(conv3_3, ksize=cnn_param.pool_window, 
                     strides=cnn_param.pool_stride, padding='SAME', name='pool3')

        conv4_1    = self.conv2d_depth_or_not(pool3,   "conv4_1", nonlinearity=tf.nn.relu)
        conv4_2    = self.conv2d_depth_or_not(conv4_1, "conv4_2", nonlinearity=tf.nn.relu)
        conv4_3    = self.conv2d_depth_or_not(conv4_2, "conv4_3", nonlinearity=tf.nn.relu)
        pool4      = tf.nn.max_pool(conv4_3, ksize=cnn_param.pool_window, 
                     strides=cnn_param.pool_stride, padding='SAME', name='pool4')

        conv5_1    = self.conv2d_depth_or_not(pool4,   "conv5_1", nonlinearity=tf.nn.relu)
        conv5_2    = self.conv2d_depth_or_not(conv5_1, "conv5_2", nonlinearity=tf.nn.relu)
        conv5_3    = self.conv2d_depth_or_not(conv5_2, "conv5_3", nonlinearity=tf.nn.relu)
    
        # feature wise convolution layers, no non-linearity
        conv_depth_1 = self.conv2d_depth_or_not(conv5_3, "conv6_1")
        # two layer of feature-wise convolution, a cubic feature transformation
        conv_depth   = self.conv2d_depth_or_not(conv_depth_1, "depth")

        # this is a replcement of last FCL layer from VGG (common in GAP & GMP models)
        # this layer does not have non-nonlinearity
        conv_last = self.conv2d_depth_or_not(conv_depth, "conv6")
        gap       = tf.reduce_mean(conv_last, [1,2])

        with tf.variable_scope("GAP"):
            gap_w = tf.get_variable("W", shape=cnn_param.layer_shapes['GAP/W'],
                    initializer=tf.random_normal_initializer(stddev=hyper.stddev))

        class_prob = tf.matmul(gap, gap_w)

        # print_model_params()
        return conv_last, gap, class_prob

    def p(self,t):
        print t.name, t.get_shape()

    def get_classmap(self, class_, conv_last):
        with tf.variable_scope("GAP", reuse=True):
            class_w = tf.gather(tf.transpose(tf.get_variable("W")), class_)
            class_w = tf.reshape(class_w, [-1, cnn_param.last_features, 1]) 
        conv_last_ = tf.image.resize_bilinear(conv_last, [hyper.image_h, hyper.image_w])
        conv_last_ = tf.reshape(conv_last_, [-1, hyper.image_h*hyper.image_w, cnn_param.last_features]) 
        classmap   = tf.reshape(tf.matmul(conv_last_, class_w), [-1, hyper.image_h,hyper.image_w])
        return classmap

