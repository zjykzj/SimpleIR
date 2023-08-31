
# 特征融合

## GAP

* GAP: Global Average Pooling, 全局平均池化；
  * 输入： 卷积层输出$X$，大小为$N\times C\times H\times W$
  * 计算： 基于通道维度求和空间特征图特征值并计算平均值
  * $N\times C\times H\times W$ -> $N\times C\times H$ -> $N\times C$
* [NIN](https://blog.zjykzj.cn/posts/359ae103.html)

## GMP

* GMP: Global Maximum Pooling, 全局最大池化；
  * 输入：卷积层输出$X$，大小为$N\times C\times H\times W$
  * 计算：基于通道维度计算空间特征图最大特征值
  * $N\times C\times H\times W$ -> $N\times C\times H$ -> $N\times C$
* [Visual Instance Retrieval with Deep Convolutional Networks](https://blog.zhujian.life/posts/841005f8.html)

## SPOC

* SPOC: Sum-Pooling of Convolutions，卷积特征求和池化；
  * 输入：卷积层输出$X$，大小为$N\times C\times H\times W$
  * 计算： 基于通道维度求和空间特征图特征值。
  * $N\times C\times H\times W$ -> $N\times C$
  * GAP vs. SPoc：差别在于计算平均值
* [Aggregating Deep Convolutional Features for Image Retrieval](https://blog.zjykzj.cn/posts/48b6e577.html)

## GEM

* GEM: Generalized-mean Pooling，广义平均池化；
  * 输入：卷积层输出$X$，大小为$N\times C\times H\times W$
  * 计算：
    * 第一步：逐个特征值求幂运算：底数为特征值，指数为超参数p（$N\times C\times H\times W$ -> $N\times C\times H\times W$）
    * 第二步：基于通道维度求和空间特征图特征值并计算平均值（$N\times C\times H\times W$ -> $N\times C$）
    * 第三步：逐个特征值求幂运算：底数为特征值，指数为1/p（$N\times C$ -> $N\times C$）
  * $N\times C\times H\times W$ -> $N\times C$
  * GAP vs. GEM：p=1时，两者相等
  * GMP vs. GEM：p=无穷大时，两者相等
* [Fine-tuning CNN Image Retrieval with No Human Annotation](https://blog.zjykzj.cn/posts/e75797c2.html) 

## R_MAC

* R_MAC: Regional Maximum Activation of Convolutions，卷积特征的区域最大激活；
  * 输入：卷积层输出$X$，大小为$N\times C\times H\times W$
  * 计算：
    * 第一步：在空间特征图上采集R个不同大小子区域
    * 第二步：逐个子区域计算，基于通道维度计算空间特征图最大特征值（$N\times C\times H_X\times W_X$ -> $N\times C\times H_X$ -> $N\times C$）
    * 第三步：逐个子区域特征计算，单独进行归一化操作
    * 第三步：求和所有子区域特征（$N\times C$ + $N\times C$ + ... + $N\times C$ -> $N\times C$）
  * $N\times C\times H\times W$ -> $N\times C$
* [Particular object retrieval with integral max-pooling of CNN activations](https://blog.zjykzj.cn/posts/47743934.html)

## CROW

* CROW: Cross-dimensional Weighting and Pooling，跨维度加权池化；
  * 输入：卷积层输出$X$，大小为$N\times C\times H\times W$ 
  * 计算：
    * 第一步：如果卷积空间特征图太大，先执行空间局部池化，使用最大池化或者平均池化
    * 第二步：计算空间加权因子spatial_weight
      * spatial_weight: 逐个空间特征点求和所有通道特征（$N\times C\times H\times W$ -> $N\times 1\times H\times W$）
      * z：逐个空间特征点执行幂运算，底数为特征值，指数为超参数spatial_a（默认为2.0）；完成后求和空间特征（$N\times 1\times H\times W$ -> $N\times 1\times 1\times 1$）
      * z：求幂运算，底数为z，指数为1/spatial_a
      * spatial_weight = (spatial_weight / z) ** (1.0 / spatial_b)（$N\times C\times H\times W$ -> $N\times 1\times H\times W$）
    * 第三步：计算通道加权因子channel_weight
      * nonzeros：逐通道计算空间特征图非零特征点的比例（$N\times C\times H\times W$ -> $N\times C$）；
      * channel_weight = torch.log(nonzeros.sum(dim=1, keepdims=True) / nonzeros)：提升稀有特征图的权重
    * 第四步：跨维度加权求和
      * 卷积特征乘以空间加权因子（$N\times C\times H\times W$ x $N\times 1\times H\times W$ -> $N\times 1\times H\times W$）
      * 全局求和池化（$N\times C\times H\times W$ -> $N\times C$）
      * 卷积特征乘以通道加权因子（$N\times C$ x $N\times C$ -> $N\times C$）
  * $N\times C\times H\times W$ -> $N\times C$
* [Cross-dimensional Weighting for Aggregated Deep Convolutional Features](https://blog.zjykzj.cn/posts/d2955233.html)