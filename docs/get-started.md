# Get started

## Use Configs

### CONFIGS

make config file like `configs/***.yaml`

### EVAL

If you want to eval only. Do it like

```shell
# specify config-path
CUDA_VISIBLE_DEVICES=0 bash tools/bash_eval.sh configs/cccf/ghostnet_100_act2_c5_cccf_224_b256_e90_g4.yaml

# specify config-path and master-port
CUDA_VISIBLE_DEVICES=0 bash tools/bash_eval.sh configs/cccf/ghostnet_100_act2_c5_cccf_224_b256_e90_g4.yaml 31226
```

**Note: for image retrieval task, loading data in single thread mode**

## Use Commands

* Extract features

```shell
python extract_features.py --model-arch resnet50 --layer fc --dataset General --image-dir data/train/ --save-dir data/gallery_fc
```

* Retrieval features

```shell
python retrieval_features.py --query-dir data/query_fc --gallery-dir data/gallery_fc --save-dir data/retrieval_fc
```

* Evaluate features

```shell
python evaluate_features.py --retrieval-dir data/retrieval_fc --retrieval-type ACCURACY
```

More info can see [RETRIEVAL](https://github.com/zjykzj/SimpleIR/tree/main/tools/retrieval)

先考虑简单的cifar10数据集，然后考虑oxford5k/paris6k这些数据集的处理

4个脚本，

1. 特征提取脚本
2. 特征检索脚本
3. 特征评估脚本
4. 特征提取、检索、评估脚本

对于图像检索，就两个库：

1. 检索库
2. 查询库

对于特征提取，数据可以不保存在本地

对于特征检索，可以不保存任何数据，反馈需要的数据

对于特征查询， 
    1. 对于准确率/精度，仅需考虑排序标签列表和查询标签列表 
    2. 对于检索mAP，需要额外考虑查询图像名

特征提取实现流程：不同提取方式
特征检索实现流程：不同检索方式
特征评估实现流程：准确率、精度、mAP
