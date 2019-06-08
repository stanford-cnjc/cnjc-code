"""
Defines CNN architectures
"""

import tensorflow as tf
from tfutils import model_tool


def alexnet_no_fc(images, train=True, norm=True, seed=0, **kwargs):
    """
    Alexnet
    """
    m = model_tool.ConvNet(seed=seed)

    conv_kwargs = {"add_bn": False, "init": "xavier", "weight_decay": 0.0001}
    pool_kwargs = {"pool_type": "maxpool"}
    fc_kwargs = {"init": "trunc_norm", "weight_decay": 0.0001, "stddev": 0.01}

    dropout = 0.5 if train else None

    m.conv(96, 11, 4, padding="VALID", layer="conv1",
           in_layer=images, **conv_kwargs)
    if norm:
        m.lrn(depth_radius=5, bias=1, alpha=0.0001, beta=0.75, layer="lrn1")
    m.pool(3, 2, layer="pool1", **pool_kwargs)

    m.conv(256, 5, 1, layer="conv2", **conv_kwargs)
    if norm:
        m.lrn(depth_radius=5, bias=1, alpha=0.0001, beta=0.75, layer="lrn2")
    m.pool(3, 2, layer="pool2", **pool_kwargs)

    m.conv(384, 3, 1, layer="conv3", **conv_kwargs)
    m.conv(384, 3, 1, layer="conv4", **conv_kwargs)

    m.conv(256, 3, 1, layer="conv5", **conv_kwargs)
    m.pool(3, 2, layer="pool5", **pool_kwargs)

    return m


def alexnet_wrapper(images, tensor_name=None, **kwargs):
    model = alexnet_no_fc(images, **kwargs)
    output = tf.get_default_graph().get_tensor_by_name(tensor_name + ":0")
    return output

