# 使用PaddlePaddle复现论文：Deep Networks with Stochastic Depth
### 基于Stochastic Depth的基于cifar10数据集的ResNet110模型
[Deep Networks with Stochastic Depth](https://arxiv.org/pdf/1603.09382v3.pdf)

## 一、简介
### 摘要：
  具有数百层的非常深的卷积网络大大减少了在竞争性基准上的错误。尽管测试时许多层的无与伦比的表现力是非常理想的，但是训练非常深的网络也带来了它自己的一系列挑战。比如梯度可能会消失，前向传播的信息通常会减少，而且训练时间可能慢得令人痛苦。要解决这些问题，论文提出了Stochastic Depth，这是一种训练过程，使得训练短网络和在测试时使用深网络这看似矛盾的设置成功。论文作者从非常深的网络开始，但在训练过程中，对于每个小批量，随机删除模型所有层的一个子集，并用 identity function绕过这些层。这种简单的方法是对残差网络（resnet系列）的最近的成功进行补充。它大大减少了训练时间，并显著改善了论文作者使用的几乎所有数据集上的测试误差。利用随机深度，论文作者说可以增加深度，即使超过1200层的剩余网络，仍可提供有意义的改善的测试误差(CIFAR-10为4.91%)。
  本项目是基于Stochastic Depth的基于cifar10数据集的ResNet110模型在 Paddle 2.x上的开源实现。该模型有3个Layer，每个Layer分别由18个BasicBlock组成，每个BasicBlock由两个conv-bn-relu和skip connection组成，其中按论文在每个mini-batch进行按照论文公式计算出的linear_decay的各block的drop_rate(论文中是保留率，1-drop_rate)一次伯努利采样，根据采样的结果决定各block保不保留，由此在训练时减小了模型平均长度，加快了训练，且测试时用full depth有模型集成的效果，提高了精度。
### 论文效果图：
![](https://github.com/zpc-666/Paddle-Stochastic-Depth-ResNet110/blob/main/images/2243.PNG)
## 二、复现精度
本次比赛的验收标准： CIFAR-10 test error=5.25 （论文指标）。我们的复现结果对比如下所示：

## 三、数据集
根据复现要求我们用的是[Cifar10](https://aistudio.baidu.com/aistudio/datasetdetail/103297)数据集。
* 数据集大小：10类别，训练集有50000张图片。测试集有10000张图片，图像大小为32x32，彩色图像；
* 数据格式：用paddle.vision.datasets.Cifar10调用，格式为cifar-10-python.tar.gz

## 四、环境依赖
* 硬件：使用了百度AI Studio平台的至尊GPU
* 框架：PaddlePaddle >= 2.0.0，平台提供了所有依赖，不必额外下载

## 五、快速开始
可以git clone https://github.com/zpc-666/Paddle-Stochastic-Depth-ResNet110到AI Studio平台去执行
cofig.py中提供了论文中提到的默认配置，故以下只按默认配置指导如何使用，如需修改参数可以直接在config.py中修改，或按argparse的用法显式地修改相应参数。

```
# 默认执行main.py，使用默认参数进行模型训练
%cd DNSD/
!bash run.sh
```
```
# 执行main.py，使用默认参数进行模型训练
%cd DNSD/
!python main.py
```
```
# 执行main.py，使用默认参数和高层API进行模型训练
%cd DNSD/
!python main.py --high_level_api True
```
```
# 执行main.py，使用高层API进行模型评估
%cd DNSD/
!python main.py --high_level_api True --mode eval --checkpoint output/model_best.pdparams
```
```
# 执行main.py，使用基础API进行模型评估
%cd DNSD/
!python main.py --mode eval --checkpoint output/model_best.pdparams
```
```
# 执行train.py，使用默认参数进行模型训练
!python train.py # --high_level_api True
```
```
# 执行eval.py，进行模型评估
%cd DNSD/
!python eval.py --checkpoint output/model_best.pdparams # --high_level_api True
```

## 六、代码结构与详细说明
  几乎完全参考https://github.com/PaddlePaddle/Contrib/wiki/Contrib-%E4%BB%A3%E7%A0%81%E6%8F%90%E4%BA%A4%E8%A7%84%E8%8C%83，参数详解见config.py每个参数的help信息。
