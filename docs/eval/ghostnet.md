# GhostNet

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
|  [ghostnet_100_act2_c5_cccf_224_b256_e90_g4](../../configs/cccf/ghostnet/ghostnet_100_act2_c5_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 48.191 / 63.768 / 70.201 / 77.878  | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |

### SCORES (Eval)

| cfg |    model   |   top1/top3/top5/top10   |   feat_type   | max_num | aggregate | enhance | distance | rank |
|:---:|:----------:|:-------------:|:----------------:|:---------:|:------------:|:-----:|:-----:|:-----:|
|  [ghostnet_100_act2_c50_cosine_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_act2_c50_cosine_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 58.102 / 72.389 / 77.354 / 83.366   | act2 |    50    |  identity |   identity  |   cosine  | normal  |
|  [ghostnet_100_conv_head_c50_cosine_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_conv_head_c50_cosine_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 58.093 / 72.380 / 77.373 / 83.375   | conv_head |    50    |  identity |   identity  |   cosine  | normal  |
|  [ghostnet_100_fc_c50_cosine_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_fc_c50_cosine_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 55.390 / 70.407 / 76.185 / 82.422   | fc |    50    |  identity |   identity  |   cosine  | normal  |
|  [ghostnet_100_conv_head_c5_cosine_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_conv_head_c5_cosine_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 54.904 / 70.341 / 75.970 / 83.207  | conv_head |    5    |  identity |   identity  |   cosine  | normal  |
|  [ghostnet_100_act2_c5_cosine_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_act2_c5_cosine_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 54.904 / 70.341 / 75.961 / 83.226  | act2 |    5    |  identity |   identity  |   cosine  | normal  |
|  [ghostnet_100_act2_c50_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_act2_c50_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 52.445 / 67.377 / 73.277 / 79.785   | act2 |    50    |  identity |   identity  |   euclidean  | normal  |
|  [ghostnet_100_conv_head_c50_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_conv_head_c50_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 52.445 / 67.377 / 73.277 / 79.776   | conv_head |    50    |  identity |   identity  |   euclidean  | normal  |
|  [ghostnet_100_act2_c60_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_act2_c60_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 52.436 / 67.415 / 73.324 / 79.719  | act2 |    60    |  identity |   identity  |   euclidean  | normal  |
|  [ghostnet_100_act2_c70_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_act2_c70_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 52.417 / 67.415 / 73.305 / 79.701  | act2 |    70    |  identity |   identity  |   euclidean  | normal  |
|  [ghostnet_100_act2_c40_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_act2_c40_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 52.258 / 67.377 / 73.268 / 79.738   | act2 |    40    |  identity |   identity  |   euclidean  | normal  |
|  [ghostnet_100_act2_c30_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_act2_c30_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 51.903 / 67.181 / 73.249 / 79.645   | act2 |    30    |  identity |   identity  |   euclidean  | normal  |
|  [ghostnet_100_fc_c50_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_fc_c50_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 51.716 / 67.546 / 73.932 / 80.888  | fc |    50    |  identity |   identity  |   euclidean  | normal  |
|  [ghostnet_100_fc_c5_cosine_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_fc_c5_cosine_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 51.697 / 67.742 / 74.596 / 81.917  | fc |    5    |  identity |   identity  |   cosine  | normal  |
|  [ghostnet_100_act2_c20_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_act2_c20_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 51.360 / 66.919 / 72.894 / 79.364   | act2 |    20    |  identity |   identity  |   euclidean  | normal  |
|  [ghostnet_100_act2_c10_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_act2_c10_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 50.229 / 65.732 / 72.052 / 79.243  | act2 |    10    |  identity |   identity  |   euclidean  | normal  |
|  [ghostnet_100_conv_head_c5_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_conv_head_c5_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 48.172 / 63.759 / 70.201 / 77.887  | conv_head |    5    |  identity |   identity  |   euclidean  | normal  |
|  [ghostnet_100_act2_c5_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_act2_c5_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 48.172 / 63.740 / 70.201 / 77.878  | act2 |    5    |  identity |   identity  |   euclidean  | normal  |
|  [ghostnet_100_fc_c5_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_fc_c5_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 47.564 / 64.778 / 71.763 / 80.112  | fc |    5    |  identity |   identity  |   euclidean  | normal  |
|  [ghostnet_100_blocks_gem_c5_cosine_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_blocks_gem_c5_cosine_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 43.385 / 59.645 / 66.227 / 73.922  | blocks |    5    |  gem |   identity  |   cosine  | normal  |
|  [ghostnet_100_blocks_gap_c5_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_blocks_gap_c5_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 41.206 / 56.802 / 63.806 / 72.099  | blocks |    5    |  gap |   identity  |   euclidean  | normal  |
|  [ghostnet_100_global_pool_c5_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_global_pool_c5_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 41.197 / 56.812 / 63.824 / 72.137  | global_pool |    5    |  identity |   identity  |   euclidean  | normal  |
|  [ghostnet_100_blocks_crow_c5_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_blocks_crow_c5_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 39.785 / 54.867 / 61.169 / 69.799  | blocks |    5    |  crow |   identity  |   euclidean  | normal  |
|  [ghostnet_100_blocks_gem_c5_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_blocks_gem_c5_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 38.308 / 54.053 / 61.178 / 70.173  | blocks |    5    |  gem |   identity  |   euclidean  | normal  |
|  [ghostnet_100_blocks_gmp_c5_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_blocks_gmp_c5_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 33.857 / 49.687 / 56.980 / 66.339  | blocks |    5    |  gmp |   identity  |   euclidean  | normal  |
|  [ghostnet_100_blocks_spoc_c5_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_blocks_spoc_c5_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 28.957 / 41.861 / 48.153 / 56.260  | blocks |    5    |  spoc |   identity  |   euclidean  | normal  |
|  [ghostnet_100_blocks_r_mac_c5_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_blocks_r_mac_c5_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 23.114 / 35.718 / 42.338 / 51.304  | blocks |    5    |  r_mac |   identity  |   euclidean  | normal  |
|  [ghostnet_100_blocks_c5_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_blocks_c5_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 6.255 / 9.696 / 11.650 / 16.568  | blocks |    5    |  identity |   identity  |   euclidean  | normal  |
