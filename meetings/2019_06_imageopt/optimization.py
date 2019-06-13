"""
Actual optimization utilities
"""

import ipdb
import tensorflow as tf
import numpy as np

from imageopt.utils.plot_utils import norm_image
import imageopt.utils.transformations as xforms

def get_optimal_image(
    model_fn,
    model_kwargs,
    checkpoint_path,
    params,
    layer_name=None,
):
    """
    Does gradient ascent to get the optimal image for a given model

    Inputs
        model_fn (fn): function to which image tensor (batch x h x w x c) can be passed
        model_kwargs (dict): other keyword arguments for the model function
        checkpoint_path (str): where to find the model checkpoint
        layer_name (str): which to layer to get image for
        params (dict): keys include
            - "channel": which channel to do optimization for
            - "learning rate"
            - "regularization"
            - "steps": how many steps to run for

    Outputs
        optimal image (224 x 224 x 3)
    """

    # set up model
    tf.reset_default_graph()
    init = tf.random_uniform_initializer(minval=0, maxval=1)
    reg = tf.contrib.layers.l2_regularizer(scale=params['regularization'])

    # preprocessing
    image_shape = (1, 128, 128, 3)
    images = tf.get_variable("images", image_shape, initializer=init, regularizer=reg)
    padded_image = xforms.pad(images, pad_amount=12)
    jittered_image = xforms.jitter(padded_image, jitter_amount=8)
    scaled_image  = xforms.random_scale(jittered_image, [1 + (i - 5) / 50. for i in range(11)])
    rotated_image = xforms.random_rotate(scaled_image, list(range(-10, 11)) + 5 * [0])
    jittered_image = xforms.jitter(rotated_image, jitter_amount=4)

    # get features for a given layer from a given model
    tensor_name = params.get('tensor_name', None)
    layer = model_fn(jittered_image, layer_name=layer_name, tensor_name=tensor_name, **model_kwargs)

    # initialize all variables except for 'images'
    sess = tf.Session()

    # extract specified channel from conv or fc layer
    if len(layer.get_shape().as_list()) == 4:
        channel = layer[0, :, :, params['channel']]
    else:
        channel = layer[0, params['channel']]

    # set up loss function
    loss_tensor = tf.negative(tf.reduce_mean(channel)) + tf.reduce_sum(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    )

    # set up optimizer
    lr_tensor = tf.constant(params['learning_rate'])

    # restrict trainable variables to the image itself
    train_vars = [
        var for var in tf.trainable_variables() if 'images' in var.name
    ]
    train_op = tf.train.AdamOptimizer(lr_tensor).minimize(loss_tensor, var_list=train_vars)

    # initialize session and all variables, restore model weights
    # sess.run(tf.initialize_variables([images]))
    sess.run(tf.global_variables_initializer())

    all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
    temp_saver = tf.train.Saver(
        var_list=[v for v in all_variables if "images" not in v.name and "beta" not in v.name]
    )
    temp_saver.restore(sess, checkpoint_path)

    ## Main Loop
    for i in range(params['steps']):
        sess.run(train_op)

    final_image = sess.run(images)
    return norm_image(final_image.squeeze())
