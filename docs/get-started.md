# Get started

* First, make config file like `configs/***.yaml`
* Second, run `train.py` like

```shell
# specify config-path
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/train.sh configs/cccf/r50_avgpool_c5_cccf_224_b256_e90_g4.yaml

# specify config-path and master-port
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/train.sh configs/cccf/r50_avgpool_c5_cccf_224_b256_e90_g4.yaml 31222
```