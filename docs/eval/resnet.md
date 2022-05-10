
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
|  [r50_avgpool_c5_cccf_224_b256_e90_g4](../../configs/cccf/resnet/r50_avgpool_c5_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 44.413 / 61.038 / 67.892 / 76.531  | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |

### SCORES (Eval)

| cfg |    model   |   top1/top3/top5/top10   |   feat_type   | max_num | aggregate | enhance | distance | rank |
|:---:|:----------:|:-------------:|:----------------:|:---------:|:------------:|:-----:|:-----:|:-----:|
|  [r50_layer4_crow_c50_cosine_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_crow_c50_cosine_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 53.950 / 68.322 / 74.381 / 81.085 | layer4 |    50    |  crow |   identity  |   cosine  | normal  |
|  [r50_layer4_crow_c20_cosine_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_crow_c20_cosine_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 53.343 / 68.079 / 74.240 / 81.197  | layer4 |    20    |  crow |   identity  |   cosine  | normal  |
|  [r50_avgpool_c50_cosine_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_avgpool_c50_cosine_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 52.959 / 68.228 / 74.128 / 81.169 | avgpool |    50    |  identity |   identity  |   cosine  | normal  |
|  [r50_avgpool_c20_cosine_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_avgpool_c20_cosine_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 52.426 / 67.779 / 73.726 / 81.103  | avgpool |    20    |  identity |   identity  |   cosine  | normal  |
|  [r50_layer4_crow_c50_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_crow_c50_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 50.098 / 65.395 / 71.678 / 78.700 | layer4 |    50    |  crow |   identity  |   euclidean  | normal  |
|  [r50_layer4_crow_c5_cosine_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_crow_c5_cosine_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 49.472 / 65.507 / 72.071 / 79.598   | layer4 |    5    |  crow |   identity  |   cosine  | normal  |
|  [r50_layer4_gem_c50_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_gem_c50_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 49.397 / 64.918 / 71.080 / 78.298 | layer4 |    50    |  gem |   identity  |   euclidean  | normal  |
|  [r50_layer4_crow_c20_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_crow_c20_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 49.294 / 64.909 / 71.248 / 78.373  | layer4 |    20    |  crow |   identity  |   euclidean  | normal  |
|  [r50_avgpool_c50_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_avgpool_c50_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 49.079 / 65.115 / 71.248 / 79.074  | avgpool |    50    |  identity |   identity  |   euclidean  | normal  |
|  [r50_layer4_gap_c50_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_gap_c50_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 49.079 / 65.115 / 71.258 / 79.084 | layer4 |    50    |  gap |   identity  |   euclidean  | normal  |
|  [r50_avgpool_c70_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_avgpool_c70_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 49.032 / 65.049 / 71.248 / 79.112 | avgpool |    70    |  identity |   identity  |   euclidean  | normal  |
|  [r50_avgpool_c60_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_avgpool_c60_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 49.023 / 65.068 / 71.258 / 79.093 | avgpool |    60    |  identity |   identity  |   euclidean  | normal  |
|  [r50_avgpool_c40_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_avgpool_c40_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 48.911 / 65.030 / 71.136 / 79.046  | avgpool |    40    |  identity |   identity  |   euclidean  | normal  |
|  [r50_avgpool_c5_cosine_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_avgpool_c5_cosine_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 48.883 / 64.741 / 71.725 / 79.448   | avgpool |    5    |  identity |   identity  |   cosine  | normal  |
|  [r50_avgpool_c30_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_avgpool_c30_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 48.658 / 64.815 / 70.827 / 78.906  | avgpool |    30    |  identity |   identity  |   euclidean  | normal  |
|  [r50_layer4_gem_c20_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_gem_c20_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 48.602 / 64.329 / 70.631 / 78.074  | layer4 |    20    |  gem |   identity  |   euclidean  | normal  |
|  [r50_layer4_gap_c20_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_gap_c20_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 48.256 / 64.404 / 70.771 / 78.644  | layer4 |    20    |  gap |   identity  |   euclidean  | normal  |
|  [r50_avgpool_c20_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_avgpool_c20_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 48.237 / 64.404 / 70.771 / 78.635  | avgpool |    20    |  identity |   identity  |   euclidean  | normal  |
|  [r50_avgpool_c10_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_avgpool_c10_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 46.489 / 62.880 / 69.547 / 77.962  | avgpool |    10    |  identity |   identity  |   euclidean  | normal  |
|  [r50_layer4_crow_c5_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_crow_c5_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 45.573 / 61.524 / 68.537 / 76.774   | layer4 |    5    |  crow |   identity  |   euclidean  | normal  |
|  [r50_layer4_gem_c5_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_gem_c5_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 44.320 / 60.701 / 67.770 / 76.064  | layer4 |    5    |  gem |   identity  |   euclidean  | normal  |
|  [r50_avgpool_c5_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_avgpool_c5_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 44.413 / 61.038 / 67.892 / 76.531   | avgpool |    5    |  identity |   identity  |   euclidean  | normal  |
|  [r50_layer4_gap_c5_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_gap_c5_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 44.413 / 61.038 / 67.892 / 76.540  | layer4 |    5    |  gap |   identity  |   euclidean  | normal  |
|  [r50_fc_c5_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_fc_c5_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 41.870 / 58.775 / 66.321 / 75.475   | fc |    5    |  identity |   identity  |   euclidean  | normal  |
|  [r50_layer4_gmp_c5_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_gmp_c5_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 41.356 / 58.065 / 65.199 / 74.296  | layer4 |    5    |  gmp |   identity  |   euclidean  | normal  |
|  [r50_layer4_spoc_c5_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_spoc_c5_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 36.765 / 52.062 / 58.943 / 68.060  | layer4 |    5    |  spoc |   identity  |   euclidean  | normal  |
|  [r50_layer4_r_mac_c5_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_r_mac_c5_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 26.526 / 42.057 / 49.331 / 60.056  | layer4 |    5    |  r_mac |   identity  |   euclidean  | normal  |
|  [r50_layer4_c5_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_layer4_c5_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 23.413 / 35.138 / 40.954 / 49.444   | layer4 |    5    |  identity |   identity  |   euclidean  | normal  |
