# Get started

## Use Configs

### CONFIGS

make config file like `configs/***.yaml`

### TRAIN

run `train.py` like

```shell
# specify config-path
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/bash_train.sh configs/cccf/ghostnet_100_act2_c5_cccf_224_b256_e90_g4.yaml

# specify config-path and master-port
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/bash_train.sh configs/cccf/ghostnet_100_act2_c5_cccf_224_b256_e90_g4.yaml 31222
```

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