
# ResNet

## About CCCF

    CCCF is a custom mixed classification dataset, including

    1. CIFAR100: https://paperswithcode.com/dataset/cifar-100
    2. CUB-200-2011: https://paperswithcode.com/dataset/cub-200-2011
    3. Caltech-101: https://paperswithcode.com/dataset/caltech-101
    4. Food-101: https://paperswithcode.com/dataset/food-101

    The classes num = 100 + 200 + 101 + 101 = 502

### SCORES (Train)

| cfg |    model   |   top1/top3/top5/top10   |       loss       | optimizer | lr-scheduler | epoch | pretrained |
|:---:|:----------:|:-------------:|:----------------:|:---------:|:------------:|:-----:|:-----:|
|  [r50_avgpool_c5_cccf_224_b256_e90_g4](../../configs/cccf/resnet/r50_avgpool_c5_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 56.087 / 72.485 / 78.927 / 85.996  | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |

### SCORES (Eval)

| cfg |    model   |   top1/top3/top5/top10   |   feat_type   | max_num | aggregate | enhance | distance | rank |
|:---:|:----------:|:-------------:|:----------------:|:---------:|:------------:|:-----:|:-----:|:-----:|
|  [r50_layer4_crow_c50_cosine_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_crow_c50_cosine_cccf_224_b256.yaml)   |  resnet50  | 53.950 / 68.322 / 74.381 / 81.085 | layer4 |    50    |  crow |   identity  |   cosine  | normal  |
|  [r50_layer4_crow_c20_cosine_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_crow_c20_cosine_cccf_224_b256.yaml)   |  resnet50  | 53.343 / 68.079 / 74.240 / 81.197  | layer4 |    20    |  crow |   identity  |   cosine  | normal  |
|  [r50_avgpool_c50_cosine_cccf_224_b256](../configs/cccf/resnet/r50_avgpool_c50_cosine_cccf_224_b256.yaml)   |  resnet50  | 67.896 / 81.230 / 85.762 / 90.528 | avgpool |    50    |  identity |   identity  |   cosine  | normal  |
|  [r50_avgpool_c20_cosine_cccf_224_b256](../configs/cccf/resnet/r50_avgpool_c20_cosine_cccf_224_b256.yaml)   |  resnet50  | 66.108 / 80.262 / 85.259 / 90.276  | avgpool |    20    |  identity |   identity  |   cosine  | normal  |
|  [r50_layer4_crow_c50_cccf_224_b256](../configs/cccf/resnet/r50_layer4_crow_c50_cccf_224_b256.yaml)   |  resnet50  | 65.965 / 79.918 / 84.532 / 89.528 | layer4 |    50    |  crow |   identity  |   euclidean  | normal  |
|  [r50_layer4_crow_c5_cosine_cccf_224_b256](../configs/cccf/resnet/r50_layer4_crow_c5_cosine_cccf_224_b256.yaml)   |  resnet50  | 61.452 / 77.111 / 82.871 / 89.072   | layer4 |    5    |  crow |   identity  |   cosine  | normal  |
|  [r50_layer4_gem_c50_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_gem_c50_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 49.397 / 64.918 / 71.080 / 78.298 | layer4 |    50    |  gem |   identity  |   euclidean  | normal  |
|  [r50_layer4_crow_c20_cccf_224_b256](../configs/cccf/resnet/r50_layer4_crow_c20_cccf_224_b256.yaml)   |  resnet50  | 64.039 / 78.490 / 83.534 / 89.095  | layer4 |    20    |  crow |   identity  |   euclidean  | normal  |
|  [r50_avgpool_c50_cccf_224_b256](../configs/cccf/resnet/r50_avgpool_c50_cccf_224_b256.yaml)   |  resnet50  | 64.790 / 79.250 / 84.226 / 89.488  | avgpool |    50    |  identity |   identity  |   euclidean  | normal  |
|  [r50_layer4_gap_c50_cccf_224_b256](../configs/cccf/resnet/r50_layer4_gap_c50_cccf_224_b256.yaml)   |  resnet50  | 64.787 / 79.259 / 84.224 / 89.488 | layer4 |    50    |  gap |   identity  |   euclidean  | normal  |
|  [r50_avgpool_c70_cccf_224_b256](../configs/cccf/resnet/r50_avgpool_c70_cccf_224_b256.yaml)   |  resnet50  | 65.262 / 79.502 / 84.381 / 89.626 | avgpool |    70    |  identity |   identity  |   euclidean  | normal  |
|  [r50_avgpool_c60_cccf_224_b256](../configs/cccf/resnet/r50_avgpool_c60_cccf_224_b256.yaml)   |  resnet50  | 65.012 / 79.418 / 84.299 / 89.577 | avgpool |    60    |  identity |   identity  |   euclidean  | normal  |
|  [r50_avgpool_c40_cccf_224_b256](../configs/cccf/resnet/r50_avgpool_c40_cccf_224_b256.yaml)   |  resnet50  | 64.250 / 78.943 / 83.934 / 89.329  | avgpool |    40    |  identity |   identity  |   euclidean  | normal  |
|  [r50_avgpool_c5_cosine_cccf_224_b256](../configs/cccf/resnet/r50_avgpool_c5_cosine_cccf_224_b256.yaml)   |  resnet50  | 60.657 / 76.566 / 82.375 / 88.911   | avgpool |    5    |  identity |   identity  |   cosine  | normal  |
|  [r50_avgpool_c30_cccf_224_b256](../configs/cccf/resnet/r50_avgpool_c30_cccf_224_b256.yaml)   |  resnet50  | 63.656 / 78.511 / 83.621 / 89.184  | avgpool |    30    |  identity |   identity  |   euclidean  | normal  |
|  [r50_layer4_gem_c20_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_gem_c20_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 48.602 / 64.329 / 70.631 / 78.074  | layer4 |    20    |  gem |   identity  |   euclidean  | normal  |
|  [r50_layer4_gap_c20_cccf_224_b256](../configs/cccf/resnet/r50_layer4_gap_c20_cccf_224_b256.yaml)   |  resnet50  | 62.695 / 77.805 / 83.060 / 88.887  | layer4 |    20    |  gap |   identity  |   euclidean  | normal  |
|  [r50_avgpool_c20_cccf_224_b256](../configs/cccf/resnet/r50_avgpool_c20_cccf_224_b256.yaml)   |  resnet50  | 62.695 / 77.805 / 83.060 / 88.887  | avgpool |    20    |  identity |   identity  |   euclidean  | normal  |
|  [r50_avgpool_c10_cccf_224_b256](../configs/cccf/resnet/r50_avgpool_c10_cccf_224_b256.yaml)   |  resnet50  | 60.063 / 75.612 / 81.414 / 87.943  | avgpool |    10    |  identity |   identity  |   euclidean  | normal  |
|  [r50_layer4_crow_c5_cccf_224_b256](../configs/cccf/resnet/r50_layer4_crow_c5_cccf_224_b256.yaml)   |  resnet50  | 57.373 / 73.294 / 79.558 / 86.403   | layer4 |    5    |  crow |   identity  |   euclidean  | normal  |
|  [r50_layer4_gem_c5_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_gem_c5_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 44.320 / 60.701 / 67.770 / 76.064  | layer4 |    5    |  gem |   identity  |   euclidean  | normal  |
|  [r50_avgpool_c5_cccf_224_b256](../configs/cccf/resnet/r50_avgpool_c5_cccf_224_b256.yaml)   |  resnet50  | 56.080 / 72.489 / 78.908 / 86.000   | avgpool |    5    |  identity |   identity  |   euclidean  | normal  |
|  [r50_layer4_gap_c5_cccf_224_b256](../configs/cccf/resnet/r50_layer4_gap_c5_cccf_224_b256.yaml)   |  resnet50  | 56.087 / 72.480 / 78.918 / 85.996  | layer4 |    5    |  gap |   identity  |   euclidean  | normal  |
|  [r50_fc_c5_cccf_224_b256](../configs/cccf/resnet/r50_fc_c5_cccf_224_b256.yaml)   |  resnet50  | 53.978 / 71.143 / 78.167 / 86.073   | fc |    5    |  identity |   identity  |   euclidean  | normal  |
|  [r50_layer4_gmp_c5_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_gmp_c5_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 41.356 / 58.065 / 65.199 / 74.296  | layer4 |    5    |  gmp |   identity  |   euclidean  | normal  |
|  [r50_layer4_spoc_c5_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_spoc_c5_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 36.765 / 52.062 / 58.943 / 68.060  | layer4 |    5    |  spoc |   identity  |   euclidean  | normal  |
|  [r50_layer4_r_mac_c5_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_r_mac_c5_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 26.526 / 42.057 / 49.331 / 60.056  | layer4 |    5    |  r_mac |   identity  |   euclidean  | normal  |
|  [r50_layer4_c5_cccf_224_b256](../configs/cccf/resnet/r50_layer4_c5_cccf_224_b256.yaml)   |  resnet50  | 31.942 / 44.299 / 50.519 / 58.777   | layer4 |    5    |  identity |   identity  |   euclidean  | normal  |
