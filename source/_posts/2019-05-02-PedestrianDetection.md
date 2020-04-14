---
title: 'Pedestrian Detection'
date: 2019-05-02
tags: [Deep Learning]
---

这篇文章是完成CSE293课程作业的步骤，主要包含GCP服务器的配置，Faster RCNN的项目使用，CityPersons数据集的配置介绍，以防自己忘记。



# GCP 搭建参考

- [GCP配置参考1](https://github.com/sdrangan/introml/blob/master/GCP/terminal.md)
- [GCP配置参考2](https://github.com/sdrangan/introml/blob/master/GCP/getting_started.md)

<!--more-->

# Faster RCNN Repo

[Faster RCNN TF版本 Github地址](https://github.com/endernewton/tf-faster-rcnn)

预训练模型在项目地址里可以下载

## 配置

- 安装requirements

- ```shell
  pip install cython
  pip install easydict
  pip install opencv-python
  ```

- 进入lib文件夹，make配置一下

  - 可能需要求改一下setup.py中的文件GPU的架构

- 我使用的是GCP的Tesla K80，因此修改成sm_37

- make 一下

- 创建一个model/，把下载的预训练的Faster RCNN model 放进这个文件夹中并解压

## 测试Demo

- 把待测试图片放入data/demo下面

- 修改tools/demo.py的内容，主要包括

  - 预训练参数的路径

  - 待测试文件名

  - plt show修改成savefig，为了能够在服务器使用plt，需要添加

    ```python
    import matplotlib
    matplotlib.use('Agg')
    ```

![test.png](https://i.loli.net/2019/05/03/5ccb46a1dc239.png)

# Dataset组织

CityPersons

[图片下载地址](https://link.jianshu.com/?t=https%3A%2F%2Fwww.cityscapes-dataset.com%2Ffile-handling%2F%3FpackageID%3D3)

[标注下载地址](https://link.jianshu.com/?t=https%3A%2F%2Fwww.cityscapes-dataset.com%2Ffile-handling%2F%3FpackageID%3D28)

[Citypersons Benchmark 地址](https://bitbucket.org/shanshanzhang/citypersons/src/default/)

## 标注文件格式转换 Json$\rightarrow$XML

[格式转换参考Repo](https://github.com/dongdonghy/citypersons2voc)

- 把所有的.json文件放入同样的annotation文件夹当中，放入annotationVOC文件夹

```shell
python ./scripts/citypersons2voc.py --input_dir "test" --output_dir "testXML"
```

```shell
numbers of images:  3475
number of ped:  19683
number of ignore:  13882
number of images with 0 labels :  210
max number of labels in single images:  99
number of heavy:  6827
number of small:  4464
number of heavy and small:  1692
min width of ped:   2.0
min height of ped:  6.0
max width of ped:   396.0
max height of ped:  965.0
mean ratio pf ped:  0.409959461309
max ratio of ped:  0.5
min ratio of ped:  0.285714285714
```

## 图片归并到同一个文件夹

解压tgz文件

```shell
tar zxvf backups.tgz
```

查看当前目录条目数

```shell
ls -l | grep "^-" | wc -l
```

- 不管是train图片还是test图片都放到同一个文件夹JPEGImages中
- 同时统计train，val，test集的图片名字到三个txt的文件当中，不需要保存后缀
  - Citypersons train: 2975 张
  - Citypersons val: 500 张
  - Citypersons test: 1525 张
- 将3个txt文件放入Imageset/Main文件夹中

## 最终数据集文件结构

![WX20190502-111743@2x.png](https://i.loli.net/2019/05/03/5ccb345349d5e.png)

## 编写数据读取接口

- 编写lib/cityperson.py

  - 仿照pascal_voc进行修改
  - 修改类名
  - name
  - self._classes
  - _get_default_path()

- 修改lib/factory.py

  - 增加cityperson的key-value

    ```python
    # Set up Citypersons_<split>
    for split in ['train', 'val', 'test']:
      name = 'cityperson_{}'.format(split)
      __sets[name] = (lambda split = split: cityperson(split))
    ```

- 修改experiments/scripts/train_faster_rcnn.sh(test_faster_rcnn.sh)

  - 增加citypersons的定义

    ```shell
    cityperson)
        TRAIN_IMDB="cityperson_train"
        TEST_IMDB="cityperson_test"
        STEPSIZE=['50000']
        ITERS=70000
        ANCHORS="[8,16,32]"
        RATIOS="[0.5,1,2]"
        ;;
    ```

  - 删除后面命令中的time

# 训练

## 准备VGG16的参数

下载VGG16的已经训练好的参数

```sh
mkdir -p data/imagenet_weights
cd data/imagenet_weights
wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xzvf vgg_16_2016_08_28.tar.gz
mv vgg_16.ckpt vgg16.ckpt
cd ../..
```

## 命令

```shell
./experiments/scripts/train_faster_rcnn.sh 0 cityperson vgg16
```

## 报错解决

1. 执行上面训练命令时候出现没有time

   solution：删除time

2. 执行训练命令出现"没有difficult"属性

   solution：在citypersons2voc.py文件中增加一个difficult节点

3. 执行训练命令中出现

   ```shell
   File "/home/guanghao0419/tf-faster-rcnn/tools/../lib/datasets/imdb.py", line 118, in append_flipped_images
       assert (boxes[:, 2] >= boxes[:, 0]).all()
   AssertionError
   ```

   从根本上解决问题，检查一下是否有下面两种情况：

   - xmin/ymin是0
   - xmax/ymax已经超过了图像的宽/高

   Solution:

   - xmin/ymin xmax/ymax都+1
   - 检查xmax和ymax超过宽/高的，赋值成宽-1/高-1

   例如下图最右侧的半身人

![ok.png](https://i.loli.net/2019/05/03/5ccb98d6a0b38.png)

4. rpn box loss is nan

   ```sh
   /home/guanghao0419/tf-faster-rcnn/tools/../lib/model/bbox_transform.py:27: RuntimeWarning: invalid value encountered in log
     targets_dw = np.log(gt_widths / ex_widths)
   iter: 80 / 70000, total loss: nan
    >>> rpn_loss_cls: 0.669006
    >>> rpn_loss_box: nan
    >>> loss_cls: 0.000197
    >>> loss_box: 0.000000
    >>> lr: 0.001000
   ```

   还是数据标注的问题，检查发现除了某些标注框小于0，有些大于宽/高

   solution: 已经在citypersons2voc.py重新对这些违法的标注框重新赋值了

# 测试

训练好的模型保存在

```
./output/vgg16/cityperson_train/default/vgg16_faster_rcnn_iter_70000.ckpt
```

## 命令

```sh
./experiments/scripts/test_faster_rcnn.sh 0 cityperson vgg16
```

## 报错解决

1. 找不到文件

```shell
IOError: [Errno 2] No such file or directory: '/path/to/my/tf-faster-rcnn-master/data/VOCdevkit2007/results/VOC2007/Main/comp4_04d40519-2eb7-49c9-aff7-db661169d037_xxxx.txt'
```

Solution：按照上面的路径创建文件夹results/VOC2010/Main

2. 不能对浮点字符串进行int的强制转换

```shell
File "/home/guanghao0419/tf-faster-rcnn/tools/../lib/datasets/voc_eval.py", line 23, in parse_rec
    obj_struct['truncated'] = int(obj.find('truncated').text)
ValueError: invalid literal for int() with base 10: '0.647058823529'
```

Solution：先修改成float 再转换成int

![WX20190509-091012@2x.png](https://i.loli.net/2019/05/10/5cd450fef2eef.png)

# 解决Occlusion问题 — Repulsion Loss

[Tensorflow版本的Faster RCNN Loss的计算位置](https://github.com/endernewton/tf-faster-rcnn/blob/master/lib/nets/network.py)

[repulsion loss的实现](https://github.com/dongdonghy/repulsion-loss-faster-rcnn-pytorch/blob/master/lib/model/faster_rcnn/repulsion_loss.py)

https://github.com/JegernOUTT/repulsion_loss/blob/master/repulsion_loss.py

https://github.com/dongdonghy/repulsion-loss-faster-rcnn-pytorch

# 参考链接

- [CSDN 如何组织数据集](https://blog.csdn.net/yaoqi_isee/article/details/79254574)

