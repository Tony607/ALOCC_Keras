
import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  """Helper function to construct a deconv "layer" with tf.nn.conv2d_transpose
  
  Arguments:
    input_ {Tensor} -- Input tensor
    output_shape {list} -- output shape, list of 4 integers.
  
  Keyword Arguments:
    k_h {int} -- kernel/filter height (default: {5})
    k_w {int} -- kernel/filter width (default: {5})
    d_h {int} -- stride for height (default: {2})
    d_w {int} -- stride for width (default: {2})
    stddev {float} -- std use initialize kernel tensor. (default: {0.02})
    name {str} -- variable_scope name (default: {"deconv2d"})
    with_w {bool} -- Whether return w, bias as well e.g. (deconv, w, biases) (default: {False})
  
  Returns:
    depends on with_w, either return deconv or (deconv, w, biases)
  """

  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv