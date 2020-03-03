---
title: Faster R-CNN源代码(TF)--networks/network.py, VGGnet_train.py, factory.py
date: 2018-08-11 22:05:40
tags: Faster R-CNN
categories: Computer Vision
---

这篇文章是对Faster R-CNN源代码中的网络结构的、目标函数的定义部分的讲解。

<!--more-->

# 参考资料

这段代码中定义了一个layer装饰器，装饰器的解释可以参考博客[Python闭包函数和装饰器练习](https://blog.csdn.net/qq_33297776/article/details/81591425)。
# network.py

&nbsp;&nbsp;&nbsp;&nbsp;这个文件定义了CNN网络的操作，卷积，Relu，maxPool等等。都被wrapper上了layer装饰器。
```python
import numpy as np
import tensorflow as tf
import roi_pooling_layer.roi_pooling_op as roi_pool_op
import roi_pooling_layer.roi_pooling_op_grad
from rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer_tf import anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer_tf import proposal_target_layer as proposal_target_layer_py

DEFAULT_PADDING = 'SAME'

#方法op的装饰器
#装饰器的大致思想就是，闭包函数饮用了外部函数的参数（在装饰器里是个函数），然后闭包函数除了调用外部函数这个参数函数以外，包装一些其他的东西，扩展这个参数函数。

#这里的layer就是一个装饰器，op是参数函数
#经过闭包函数layer_decorated包装了一些其他的功能后，返回这个layer_decorated函数，实际上就是返回了一个升级版的op
def layer(op):
    def layer_decorated(self, *args, **kwargs):
        #获取layer的名字，调用get_unique_name生成一个唯一名字作为字典的key
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # 取出输入，如果inputs没有内容，则报错，否则就取出
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
            
        #被装饰函数op在这里被调用，取出的输入传进去，并且给入其他的参数
        layer_output = op(self, layer_input, *args, **kwargs)
        #将该层的输入传入到look-up-table里，以便查找
        self.layers[name] = layer_output
        #调用feed方法将layer_output给入，目的是返回self进行链式调用
        self.feed(layer_output)
        #返回self
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        #被子类重载
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, saver, ignore_missing=False):
        if data_path.endswith('.ckpt'):
            saver.restore(session, data_path)
        else:
            data_dict = np.load(data_path).item()
            for key in data_dict:
                with tf.variable_scope(key, reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print "assign pretrain model "+subkey+ " to "+key
                        except ValueError:
                            print "ignore "+key
                            if not ignore_missing:

                                raise

    #返回self的目的就是能够进行链式调用，其实只是看起来美观
    #实际上每一个函数都没有返回值
    #相当于就是
    #self.func1()
    #self.func2()
    #...
    #self.funcn()
    #写法好看而已
    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                    print layer
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        #返回某一个层的输出
        try:
            layer = self.layers[layer]
        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        #这个函数是用来生成layer唯一的名字的
        #例如conv层，可能有conv1,conv2,conv3..。
        #输入prefix就是conv
        #dict.items()返回的是可迭代的类型，例如{'a':1,'b':2}，那么返回 dict_items([('a', 1), ('b', 2)])
        #利用即查看有没有conv开头的key，记录有的个数（true），相加再加1为id
        id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True):
        #建立一个tensorflow的变量
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)

    def validate_padding(self, padding):
        #判断padding类型是否符合要求
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=1, trainable=True):
        #检查padding类型是否符合要求
        #输入:k_h 核高，k_w 核宽，c_o 应该就是核数，s_h h方向上滑动距离 s_w w方向上滑动距离
        self.validate_padding(padding)

        #input形状为[batch, in_height, in_width, in_channels]
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, trainable)
            biases = self.make_var('biases', [c_o], init_biases, trainable)

            if group==1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)
            if relu:
                bias = tf.nn.bias_add(conv, biases)
                return tf.nn.relu(bias, name=scope.name)
            return tf.nn.bias_add(conv, biases, name=scope.name)

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def roi_pool(self, input, pooled_height, pooled_width, spatial_scale, name):
        # only use the first input
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        if isinstance(input[1], tuple):
            input[1] = input[1][0]

        print input
        return roi_pool_op.roi_pool(input[0], input[1],
                                    pooled_height,
                                    pooled_width,
                                    spatial_scale,
                                    name=name)[0]

    @layer
    def proposal_layer(self, input, _feat_stride, anchor_scales, cfg_key, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        return tf.reshape(tf.py_func(proposal_layer_py,[input[0],input[1],input[2], cfg_key, _feat_stride, anchor_scales], [tf.float32]),[-1,5],name =name)


    @layer
    def anchor_target_layer(self, input, _feat_stride, anchor_scales, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:

            rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = tf.py_func(anchor_target_layer_py,[input[0],input[1],input[2],input[3], _feat_stride, anchor_scales],[tf.float32,tf.float32,tf.float32,tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels')
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets')
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights')
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights')


            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


    @layer
    def proposal_target_layer(self, input, classes, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        with tf.variable_scope(name) as scope:

            rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights = tf.py_func(proposal_target_layer_py,[input[0],input[1],classes],[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])

            rois = tf.reshape(rois,[-1,5] , name = 'rois') 
            labels = tf.convert_to_tensor(tf.cast(labels,tf.int32), name = 'labels')
            bbox_targets = tf.convert_to_tensor(bbox_targets, name = 'bbox_targets')
            bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name = 'bbox_inside_weights')
            bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name = 'bbox_outside_weights')

           
            return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


    @layer
    def reshape_layer(self, input, d,name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob_reshape':
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
                    int(d),tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),input_shape[2]]),[0,2,3,1],name=name)
        else:
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
                    int(d),tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),input_shape[2]]),[0,2,3,1],name=name)

    @layer
    def feature_extrapolating(self, input, scales_base, num_scale_base, num_per_octave, name):
        return feature_extrapolating_op.feature_extrapolating(input,
                              scales_base,
                              num_scale_base,
                              num_per_octave,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable)
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
        else:
            return tf.nn.softmax(input,name=name)

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)

```

#VGGnet_train.py
&nbsp;&nbsp;&nbsp;&nbsp;这个文件定义了VGGnet_train这个类，继承了network类，利用父类的基本操作，建立了VGG网络的网络结构。
```python
import tensorflow as tf
from networks.network import Network

n_classes = 21
_feat_stride = [16,]
anchor_scales = [8, 16, 32]

class VGGnet_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes})
        self.trainable = trainable
        self.setup()

        # create ops and placeholders for bbox normalization process
        with tf.variable_scope('bbox_pred', reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")

            self.bbox_weights = tf.placeholder(weights.dtype, shape=weights.get_shape())
            self.bbox_biases = tf.placeholder(biases.dtype, shape=biases.get_shape())

            self.bbox_weights_assign = weights.assign(self.bbox_weights)
            self.bbox_bias_assign = biases.assign(self.bbox_biases)

    def setup(self):
    """
    注意这个位置的链式调用，仔细分析一下其用法:
    feed()的返回值是对象实例self，而conv()的返回值是tf.nn结果。
    But!!!conv被layer装饰器wrapper了一下，在wrapper中最后几行代码：
    layer_output = op(self, layer_input, *args, **kwargs)
    self.layers[name] = layer_output
    self.feed(layer_output)
    return self
    能够看到conv的结果被放到了layer中，并且给到了feed函数，而feed函数又会把这个结果方到inputs里，这时候返回self。
    由于返回的是self，那么可以链式进行下一次调用。在wrapper的开始位置又会从inputs中取出layer_input，所以我们在看到下面的conv()调用中没有输入input，但实际上是在wrapper的调用中给出了。
    """
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
             .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
             .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3'))
        #========= RPN ============
        (self.feed('conv5_3')
             .conv(3,3,512,1,1,name='rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*3*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score'))

        (self.feed('rpn_cls_score','gt_boxes','im_info','data')
             .anchor_target_layer(_feat_stride, anchor_scales, name = 'rpn-data' ))

        # Loss of rpn_cls & rpn_boxes

        (self.feed('rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*3*4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred'))

        #========= RoI Proposal ============
        (self.feed('rpn_cls_score')
             .reshape_layer(2,name = 'rpn_cls_score_reshape')
             .softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
             .reshape_layer(len(anchor_scales)*3*2,name = 'rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','im_info')
             .proposal_layer(_feat_stride, anchor_scales, 'TRAIN',name = 'rpn_rois'))

        (self.feed('rpn_rois','gt_boxes')
             .proposal_target_layer(n_classes,name = 'roi-data'))


        #========= RCNN ============
        (self.feed('conv5_3', 'roi-data')
             .roi_pool(7, 7, 1.0/16, name='pool_5')
             .fc(4096, name='fc6')
             .dropout(0.5, name='drop6')
             .fc(4096, name='fc7')
             .dropout(0.5, name='drop7')
             .fc(n_classes, relu=False, name='cls_score')
             .softmax(name='cls_prob'))

        (self.feed('drop7')
             .fc(n_classes*4, relu=False, name='bbox_pred'))

```
#factory.py
&nbsp;&nbsp;&nbsp;&nbsp;定义了一个网络，方便进行调用。如果想使用VGGnet_train，那么便能够方便调用VGGnet_train或者是VGGnet_test。
```python
# --------------------------------------------------------
# SubCNN_TF
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------
"""Factory method for easily getting imdbs by name."""
__sets = {}
import networks.VGGnet_train
import networks.VGGnet_test
import pdb
import tensorflow as tf

#__sets['VGGnet_train'] = networks.VGGnet_train()
#__sets['VGGnet_test'] = networks.VGGnet_test()

def get_network(name):
	#给定一个网络的名字，调用该网络的对象实例。
	#实际上下面这段代码表达只有两个VGGnet_train和VGGnet_test两个网络
    #if not __sets.has_key(name):
    #    raise KeyError('Unknown dataset: {}'.format(name))
    #return __sets[name]
    if name.split('_')[1] == 'test':
       return networks.VGGnet_test()
    elif name.split('_')[1] == 'train':
       return networks.VGGnet_train()
    else:
       raise KeyError('Unknown dataset: {}'.format(name))
    
def list_networks():
	#列出所有网络的名字
    """List all registered imdbs."""
    return __sets.keys()
```