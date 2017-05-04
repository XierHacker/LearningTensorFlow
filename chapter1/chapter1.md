更新：2017.2.27

>因为tensorflow1.0出来了，很多不兼容，所以这部分更新了一下。以适应tensorflow1.0

# 一.tensorflow背景
背景直接不多说了，一般学习已经开始学习tensorflow的都应该都知道tensorflow的背景了。所以这里直接略过啦。

# 二.安装
在之前的博客里面有详细的tensorflow的安装和配置。相对caffe来说，还是比较简单的。不熟悉的看这里 [***tensorflow安装***](http://blog.csdn.net/xierhacker/article/details/53035989)
# 三.编程思想

这里直接翻译的官方文档的介绍，TensorFlow 使用***图***来表示计算任务. 图中的节点被称之为 op (operation 的缩写). 一个 op获得 0 个或多个 Tensor , 执行计算, 产生 0 个或多个 Tensor . 每个 Tensor 是一个类型化的多维数组.tensor也是tensorflow中的核心数据类型。

一个 TensorFlow 图（graph）描述了计算的过程. 为了进行计算, 图必须在会话（session）里被启动. 会话将图的op分发到诸如 CPU 或 GPU 之类的 设备 上, 同时提供执行 op 的方法. 这些方法执行后, 将产生的 tensor 返回.

TensorFlow 程序通常被组织成一个**构建阶段**和一个**执行阶段**.

>在构建阶段, op 的执行步骤被描述成一个图.
在执行阶段, 使用会话执行执行图中的op.例如,通常在构建阶段创建一个图来表示和训练神经网络,然后在执行阶段反复执行图中的训练 op.



# 四.基本例子

前面说了那么多是很抽象的，这里给出一个基本例子，这个例子现在都可以不用懂其中的含义。现在你只要知道，这个例子能够跑出结果，和这创建两个例子最基本的流程就行了。后面的文章会详细分析。

首先要放在这里的例子是超级经典的Hello World啦，没有hello world的教程都是耍流氓对吧。


```python
#import tensorflow
from __future__ import print_function,division
import tensorflow as tf

#define the graph
info_op=tf.constant("hello,world")
a=tf.constant(10)
b=tf.constant(20)
add_op=tf.add(a,b)

#run graph in session
with tf.Session() as session:
    print(session.run(info_op))
    print(session.run(add_op))
```
结果：

![这里写图片描述](http://img.blog.csdn.net/20161109161921494)

这里需要把一些出现了的你可能会迷糊的“关键字”挑出来。分别是`constant`，`tf.add`，`tf.Session`这些。但是你现在并不需要马上理解他们是什么。挑出来的原因就是这些事调用的tensorflow的API，你现在不用知道是什么，虽然你可能已经猜出来他们的意义了。
这里想要说的重点是建立一个tensorflow程序的过程。首先肯定是载入必要的包，这是废话。然后创建图（graph），然后再在session里面运行图。如果现在还是很晕的话，没有关系，例子见多了就熟悉了。

最后，插入一个tensoflow的API官方文档，以后的内容会随时链接到这里。
[API r1.0](https://www.tensorflow.org/api_docs/python/)