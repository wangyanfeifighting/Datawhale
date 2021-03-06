五、模型融合
Tip:此部分为零基础入门数据挖掘的 Task5 模型融合 部分，带你来了解各种模型结果的融合方式，在比赛的攻坚
时刻冲刺Top，欢迎大家后续多多交流。
赛题：零基础入门数据挖掘 - 二手车交易价格预测
地址：https://tianchi.aliyun.com/competition/entrance/231784/introduction?
spm=5176.12281957.1004.1.38b02448ausjSX
(https://tianchi.aliyun.com/competition/entrance/231784/introduction?
spm=5176.12281957.1004.1.38b02448ausjSX
5.1 模型融合目标
对于多种调参完成的模型进行模型融合。
完成对于多种模型的融合，提交融合结果并打卡。
5.2 内容介绍
模型融合是比赛后期一个重要的环节，大体来说有如下的类型方式。
1. 简单加权融合:
回归（分类概率）：算术平均融合（Arithmetic mean），几何平均融合（Geometric mean）；
分类：投票（Voting)
综合：排序融合(Rank averaging)，log融合
2. stacking/blending:
构建多层模型，并利用预测结果再拟合预测。
4. boosting/bagging（在xgboost，Adaboost,GBDT中已经用到）:
多树的提升方法

5.3 Stacking相关理论介绍
1) 什么是 stacking

简单来说 stacking 就是当用初始训练数据学习出若干个基学习器后，将这几个学习器的预测结果作为新的训练
集，来学习一个新的学习器。
将个体学习器结合在一起的时候使用的方法叫做结合策略。对于分类问题，我们可以使用投票法来选择输出最多
的类。对于回归问题，我们可以将分类器输出的结果求平均值。
上面说的投票法和平均法都是很有效的结合策略，还有一种结合策略是使用另外一个机器学习算法来将个体机器
学习器的结果结合在一起，这个方法就是Stacking。
在stacking方法中，我们把个体学习器叫做初级学习器，用于结合的学习器叫做次级学习器或元学习器（metalearner），
次级学习器用于训练的数据叫做次级训练集。次级训练集是在训练集上用初级学习器得到的。
2) 如何进行 stacking

3）Stacking的方法讲解
首先，我们先从一种“不那么正确”但是容易懂的Stacking方法讲起。
Stacking模型本质上是一种分层的结构，这里简单起见，只分析二级Stacking.假设我们有2个基模型 Model1_1、
Model1_2 和 一个次级模型Model2
Step 1. 基模型 Model1_1，对训练集train训练，然后用于预测 train 和 test 的标签列，分别是P1，T1
Model1_1 模型训练:
训练后的模型 Model1_1 分别在 train 和 test 上预测，得到预测标签分别是P1，T1
Step 2. 基模型 Model1_2 ，对训练集train训练，然后用于预测train和test的标签列，分别是P2，T2
Model1_2 模型训练:
训练后的模型 Model1_2 分别在 train 和 test 上预测，得到预测标签分别是P2，T2
Step 3. 分别把P1,P2以及T1,T2合并，得到一个新的训练集和测试集train2,test2.
再用 次级模型 Model2 以真实训练集标签为标签训练,以train2为特征进行训练，预测test2,得到最终的测试集预测
的标签列 。

这就是我们两层堆叠的一种基本的原始思路想法。在不同模型预测的结果基础上再加一层模型，进行再训练，从
而得到模型最终的预测。
Stacking本质上就是这么直接的思路，但是直接这样有时对于如果训练集和测试集分布不那么一致的情况下是有
一点问题的，其问题在于用初始模型训练的标签再利用真实标签进行再训练，毫无疑问会导致一定的模型过拟合
训练集，这样或许模型在测试集上的泛化能力或者说效果会有一定的下降，因此现在的问题变成了如何降低再训
练的过拟合性，这里我们一般有两种方法。
1. 次级模型尽量选择简单的线性模型
2. 利用K折交叉验证