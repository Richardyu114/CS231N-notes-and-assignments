# cs231n training camp 

## 课程资料
1. [课程主页](http://cs231n.stanford.edu/)  
2. [英文笔记](http://cs231n.github.io/)  
3. [中文笔记](https://zhuanlan.zhihu.com/p/21930884)  
4. [课程视频](https://www.bilibili.com/video/av13260183?p=1)  
5. [课程大纲](http://cs231n.stanford.edu/syllabus.html)

## 知识工具

### 数学工具
#### cs229资料：
- [线性代数](http://web.stanford.edu/class/cs224n/readings/cs229-linalg.pdf)  
- [概率论](http://web.stanford.edu/class/cs224n/readings/cs229-prob.pdf)  
- [凸函数优化](http://web.stanford.edu/class/cs224n/readings/cs229-cvxopt.pdf)  
- [随机梯度下降算法](http://cs231n.github.io/optimization-1/)  

#### 中文资料：    
- [机器学习中的数学基本知识](https://www.cnblogs.com/steven-yang/p/6348112.html)  
- [统计学习方法](http://vdisk.weibo.com/s/vfFpMc1YgPOr)  

**大学数学课本（从故纸堆里翻出来^_^）**  

### 编程工具 
- [Python复习](http://web.stanford.edu/class/cs224n/lectures/python-review.pdf)  
- [PyTorch教程](https://www.udacity.com/course/deep-learning-pytorch--ud188)  
- [廖雪峰python3教程](https://www.liaoxuefeng.com/article/001432619295115c918a094d8954bd493037b03d27bf9a9000)
- [github教程](https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000)
- [深度学习的学习路线](https://github.com/L1aoXingyu/Roadmap-of-DL-and-ML/blob/master/README_cn.md)和[开源深度学习课程](http://www.deeplearningweekly.com/blog/open-source-deep-learning-curriculum/)
- [mxnet/gluon 教程](https://zh.gluon.ai/)
- [我的知乎专栏](https://zhuanlan.zhihu.com/c_94953554)和[pytorch教程](https://github.com/L1aoXingyu/code-of-learn-deep-learning-with-pytorch)
- [官方pytorch教程](https://pytorch.org/tutorials/)和一个比较好的[教程](https://github.com/yunjey/pytorch-tutorial)
- [tensorflow教程](https://github.com/aymericdamien/TensorFlow-Examples)

### 作业详解

- [**solutions**](https://github.com/Richardyu114/CS231N-notes-and-assignments/tree/master/solutions)


## 前言
对于算法工程师，不同的人的认知角度都是不同的，我们通过下面三个知乎的高票回答帮助大家了解算法工程师到底需要做什么样的事，工业界需要什么样的能力

[从今年校招来看，机器学习等算法岗位应届生超多，竞争激烈，未来 3-5 年机器学习相关就业会达到饱和吗?](https://www.zhihu.com/question/66406672/answer/317489657)

[秋招的 AI 岗位竞争激烈吗?](https://www.zhihu.com/question/286925266/answer/491117602)

[论算法工程师首先是个工程师之深度学习在排序应用踩坑总结](https://zhuanlan.zhihu.com/p/44315278)



## 教程
## 学习安排
每周具体时间划分为4个部分：
- 1部分安排在周一到周二
- 2部分安排在周四到周五
- 3部分安排在周日
- 4部分作业是任何有空的时间自行完成，可以落后于学习进度
- 周三和周六休息 ^_^

### Week 1
1. 了解计算机视觉综述，历史背景和课程大纲
- 观看视频 lecture01

2. 学习数据驱动的方法, 理解 KNN 算法，初步学习线性分类器
- 观看lecture02

- 学习 [图像分类笔记上](https://zhuanlan.zhihu.com/p/20894041?refer=intelligentunit)  
- 学习 [图像分类笔记下](https://zhuanlan.zhihu.com/p/20900216)
- 学习 [线性分类笔记上](https://zhuanlan.zhihu.com/p/20918580?refer=intelligentunit)

3. 掌握本门课 python 编程的基本功
- 阅读 python 和 numpy [教程](https://zhuanlan.zhihu.com/p/20878530?refer=intelligentunit)和[代码](https://github.com/sharedeeply/cs231n-camp/blob/master/tutorial/python_numpy_tutorial.ipynb)

4. 作业   
- (热身)写一个矩阵的类，实现矩阵乘法，只能使用 python 的类(class)和列表(list)
- 完成assignment1 中的 `knn.ipynb`
- 作业详解：[knn.md](https://github.com/Richardyu114/CS231N-notes-and-assignments/blob/master/solutions/knn.md)


### Week2
1. 深入理解线性分类器的原理 
- 观看lecture03
- 学习[线性分类笔记中](https://zhuanlan.zhihu.com/p/20945670?refer=intelligentunit)
- 学习[线性分类笔记下](https://zhuanlan.zhihu.com/p/21102293)

2. 学习损失函数以及梯度下降的相关知识
- 观看lecture03
- 学习[最优化笔记](https://zhuanlan.zhihu.com/p/21360434?refer=intelligentunit)

3. 掌握矩阵求导的基本方法
- 根据[资料](https://zhuanlan.zhihu.com/p/25063314)，学习矩阵求导的基本技巧，看多少内容取决于个人需要

4. 作业
- 简述 KNN 和线性分类器的优劣, 打卡上传知知识圈
- 完成assignment1 中 `svm.ipynb`
- 作业详解：[svm.md](https://github.com/Richardyu114/CS231N-notes-and-assignments/blob/master/solutions/svm.md)


### Week3
1. 学习掌握深度学习的基石: 反向传播算法 
- 观看lecture04
- 学习[反向传播算法的笔记](https://zhuanlan.zhihu.com/p/21407711?refer=intelligentunit)

2. 理解神经网络的结构和原理
- 观看lecture04

3. 深入理解反向传播算法
- 阅读反向传播算法的[数学补充](http://cs231n.stanford.edu/handouts/derivatives.pdf)和[例子](http://cs231n.stanford.edu/handouts/linear-backprop.pdf) 

4. 作业
- 完成 assignment1 中的 `softmax.ipynb`
- 完成 assignment1 中的 `two_layer_net.ipynb`
- 作业详解1：[Softmax.md](https://github.com/Richardyu114/CS231N-notes-and-assignments/blob/master/solutions/Softmax.md)
- 作业详解2：[two_layer_net.md](https://github.com/Richardyu114/CS231N-notes-and-assignments/blob/master/solutions/two_layer_net.md)

### Week4
1. 掌握 PyTorch 中的基本操作
- 学习 pytorch 的[入门基础](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) 

2. 了解 kaggle 比赛的流程，并完成第一次的成绩提交
- 了解比赛[房价预测](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- 学习[模板代码](https://github.com/L1aoXingyu/kaggle-house-price)

3. 学习深度学习的系统[项目模板](https://github.com/L1aoXingyu/PyTorch-Project-Template)

4. 作业
- 完成 assignment1 中的 `features.ipynb`
- 修改房价预测的代码，在知识圈上提交 kaggle 的成绩
- 作业详解：[features.md](https://github.com/Richardyu114/CS231N-notes-and-assignments/blob/master/solutions/features.md)


### Week5
1. 理解 CNN 中的卷积
- 观看lecture05

2. 理解 CNN 中的 pooling
- 观看lecture05
- 学习[卷积神经网络笔记](https://zhuanlan.zhihu.com/p/22038289?refer=intelligentunit)

3. 完成 CNN 的第一个应用练习，人脸关键点检测
- 阅读 facial keypoint [小项目](https://github.com/udacity/P1_Facial_Keypoints)
- [参考代码](https://github.com/L1aoXingyu/P1_Facial_Keypoints)

4. 作业
- 思考一下卷积神经网络对比传统神经网络的优势在哪里？为什么更适合处理图像问题，知识圈打卡上传
- 完成 assignment2 中 `FullyConnectedNets.ipynb`
- 作业详解：[FullyConnectedNets1.md](https://github.com/Richardyu114/CS231N-notes-and-assignments/blob/master/solutions/FullyConnectedNets1.md)
- 作业详解：[FullyConnectedNets2.md](https://github.com/Richardyu114/CS231N-notes-and-assignments/blob/master/solutions/FullyConnectedNets2.md)


### Week6
1. 理解激活函数，权重初始化，batchnorm 对网络训练的影响
- 观看lecture06
- 学习[神经网络笔记1](https://zhuanlan.zhihu.com/p/21462488?refer=intelligentunit)

2. 深入理解 BatchNormalization
- 观看lecture06
- 学习[神经网络笔记2](https://zhuanlan.zhihu.com/p/21560667?refer=intelligentunit)

3. 总结回顾和理解深度学习中 normalize 的技巧
- 阅读文章 [深度学习中的 normalization 方法](https://zhuanlan.zhihu.com/p/33173246)

4. 作业
- 完成 assignment2 中 `BatchNormalization.ipynb`
- 完成 assignment2 中 `Dropout.ipynb`
- 作业详解：[BatchNormalization.md](https://github.com/Richardyu114/CS231N-notes-and-assignments/blob/master/solutions/BatchNormalization.md)
- 作业详解：[Dropout.md](https://github.com/Richardyu114/CS231N-notes-and-assignments/blob/master/solutions/FullyConnectedNets2.md)


### Week7
1. 理解更 fancy 的优化方法，更多的 normalize 以及正则化和迁移学习对网络训练的影响
- 观看lecture07
- 学习[神经网络笔记3](https://zhuanlan.zhihu.com/p/21741716?refer=intelligentunit)

2. 了解第二次的 kaggle 比赛 cifar10 分类
- 报名 cifar10 [比赛](https://www.kaggle.com/c/cifar-10)
- 学习[模板代码](https://github.com/L1aoXingyu/kaggle-cifar10)

3. 全面的理解深度学习中的优化算法 
- 阅读优化算法的[笔记](https://zhuanlan.zhihu.com/p/22252270)

4. 作业
- 完成 assignment2 中 `ConvolutionNetworks.ipynb`
- 修改 cifar10 的网络结构，在知识圈上提交 kaggle 成绩
- 作业详解：[ConvolutionNetworks](https://github.com/Richardyu114/CS231N-notes-and-assignments/blob/master/solutions/ConvolutionNetworks.md)

### Week8
1. 了解主流深度学习框架之间的区别与联系   
- 观看lecture08

2. 了解经典的网络结构
- 观看lecture09

3. 理解卷积神经网络的最新进展
- 学习笔记[变形卷积核、可分离卷积？卷积神经网络中十大拍案叫绝的操作](https://zhuanlan.zhihu.com/p/28749411)

4. 作业  
- 完成 assignment2 中的 `PyTorch.ipynb` 
- 学习[模板代码](https://github.com/L1aoXingyu/kaggle-plant-seeding), 尝试更大的网络结构完成 kaggle 比赛[种子类型识别](https://www.kaggle.com/c/plant-seedlings-classification)的比赛，在知识圈上提交 kaggle 成绩
- 作业详解：[Pytorch.md](https://github.com/Richardyu114/CS231N-notes-and-assignments/blob/master/solutions/PyTorch.md)


### Week9
1. 掌握 RNN 和 LSTM 的基本知识 
- 观看lecture10

2. 了解语言模型和 image caption 的基本方法
- 观看lecture10

3. 更深入的理解循环神经网络的内部原理
- 学习blog [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), [中文版本](https://www.jianshu.com/p/9dc9f41f0b29)

4. 作业
- 完成 assignment3 中的 `RNN_Captioning.ipynb` 
- 完成 assignment3 中的 `LSTM_Captioning.ipynb` 
- 完成 coco 数据集上的 [image caption 小项目](https://github.com/udacity/CVND---Image-Captioning-Project)，[参考代码](https://github.com/L1aoXingyu/image-caption-project)
- 作业详解：[RNN_Captioning.md](https://github.com/Richardyu114/CS231N-notes-and-assignments/blob/master/solutions/RNN_Captioning.md)


### Week10
1. 学习计算机视觉中的语义分割问题
- 观看lecture11

2. 学习计算机视觉中的目标检测问题 
- 观看lecture11

3. 了解目标检测中的常见算法
- 学习目标检测的[笔记](https://blog.csdn.net/v_JULY_v/article/details/80170182#commentBox)

4. 作业
- 阅读论文 [Fully Convolutional Networks for Semantic Segmentation](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) 和[中文笔记](https://zhuanlan.zhihu.com/p/30195134)
- (可选) FCN 的[复现代码](https://github.com/L1aoXingyu/fcn.pytorch)理解

### Week11
1. 理解卷积背后的原理 
- 观看lecture12

2. 学习 deep dream 和 风格迁移等有趣应用
- 观看lecture12

3. 了解无监督学习和生成模型
- 观看lecture13

4. 作业
- 完成 assignment3 中的 `NetworkVisualization-PyTorch.ipynb`
- 阅读论文 [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) 和一个详细的[讲解](https://docs.google.com/presentation/d/1rtfeV_VmdGdZD5ObVVpPDPIODSDxKnFSU0bsN_rgZXc/pub?start=false&loop=false&delayms=3000&slide=id.g178a005570_0_20175)
- (可选) SSD 的[复现代码](https://github.com/L1aoXingyu/ssd.pytorch)理解

### Week12
1. 掌握自动编码器和生成对抗网络的基本原理
- 观看lecture13

2. 了解强化学习的基本概念
- 观看lecture14

3. 学习强化学习中的 q learning 和 actor-critic 算法
- 观看lecture14

4. 作业
- 完成 assignment3 中的 `GANs-PyTorch.ipynb`
- 完成 assignment3 中的 `StyleTransfer-PyTorch.ipynb`
