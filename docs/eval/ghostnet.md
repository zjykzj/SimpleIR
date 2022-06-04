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
|  [ghostnet_100_act2_c5_cccf_224_b256_e90_g4](../../configs/cccf/ghostnet/ghostnet_100_act2_c5_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 52.885 / 69.547 / 76.293 / 83.899  | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |

### SCORES (Eval)

| cfg |    model   |   top1/top3/top5/top10   |   feat_type   | max_num | aggregate | enhance | distance | rank | re_rank | index_mode |
|:---:|:----------:|:-------------:|:----------------:|:---------:|:------------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|  [ghostnet_100_act2_c50_cosine_qe_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_act2_c50_cosine_qe_cccf_224_b256.yaml)   |  ghostnet_100  | 67.106 / 77.567 / 80.346 / 84.507   | act2 |    50    |  identity |   identity  |   cosine  | normal  |   qe  |   0  |
|  [ghostnet_100_act2_c50_cosine_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_act2_c50_cosine_cccf_224_b256.yaml)   |  ghostnet_100  | 66.809 / 80.152 / 84.811 / 89.895   | act2 |    50    |  identity |   identity  |   cosine  | normal  |   identity  |   0  |
|  [ghostnet_100_conv_head_c50_cosine_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_conv_head_c50_cosine_cccf_224_b256.yaml)   |  ghostnet_100  | 66.809 / 80.152 / 84.811 / 89.892   | conv_head |    50    |  identity |   identity  |   cosine  | normal  |   identity  |   0  |
|  [ghostnet_100_fc_c50_cosine_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_fc_c50_cosine_cccf_224_b256.yaml)   |  ghostnet_100  | 63.932 / 78.560 / 83.630 / 89.278   | fc |    50    |  identity |   identity  |   cosine  | normal  |   identity  |   0  |
|  [ghostnet_100_act2_c70_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_act2_c70_cccf_224_b256.yaml)   |  ghostnet_100  | 62.263 / 76.875 / 82.288 / 88.121  | act2 |    70    |  identity |   identity  |   euclidean  | normal  |   identity  |   0  |
|  [ghostnet_100_act2_c60_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_act2_c60_cccf_224_b256.yaml)   |  ghostnet_100  | 62.127 / 76.725 / 82.155 / 88.008  | act2 |    60    |  identity |   identity  |   euclidean  | normal  |   identity  |   0  |
|  [ghostnet_100_conv_head_c50_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_conv_head_c50_cccf_224_b256.yaml)   |  ghostnet_100  | 61.692 / 76.489 / 82.006 / 87.955   | conv_head |    50    |  identity |   identity  |   euclidean  | normal  |   identity  |   0  |
|  [ghostnet_100_act2_c50_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_act2_c50_cccf_224_b256.yaml)   |  ghostnet_100  | 61.683 / 76.491 / 82.003 / 87.952   | act2 |    50    |  identity |   identity  |   euclidean  | normal  |   identity  |   0  |
|  [ghostnet_100_act2_c40_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_act2_c40_cccf_224_b256.yaml)   |  ghostnet_100  | 61.248 / 76.178 / 81.896 / 87.924   | act2 |    40    |  identity |   identity  |   euclidean  | normal  |   identity  |   0  |
|  [ghostnet_100_fc_c50_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_fc_c50_cccf_224_b256.yaml)   |  ghostnet_100  | 61.171 / 76.727 / 82.485 / 88.558  | fc |    50    |  identity |   identity  |   euclidean  | normal  |   identity  |   0  |
|  [ghostnet_100_act2_c30_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_act2_c30_cccf_224_b256.yaml)   |  ghostnet_100  | 60.580 / 75.680 / 81.545 / 87.758   | act2 |    30    |  identity |   identity  |   euclidean  | normal  |   identity  |   0  |
|  [ghostnet_100_conv_head_c5_cosine_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_conv_head_c5_cosine_cccf_224_b256.yaml)   |  ghostnet_100  | 59.965 / 76.061 / 82.071 / 88.796  | conv_head |    5    |  identity |   identity  |   cosine  | normal  |   identity  |   0  |
|  [ghostnet_100_act2_c5_cosine_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_act2_c5_cosine_cccf_224_b256.yaml)   |  ghostnet_100  | 59.958 / 76.066 / 82.069 / 88.799  | act2 |    5    |  identity |   identity  |   cosine  | normal  |   identity  |   0  |
|  [ghostnet_100_act2_c20_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_act2_c20_cccf_224_b256.yaml)   |  ghostnet_100  | 59.341 / 74.930 / 80.860 / 87.237   | act2 |    20    |  identity |   identity  |   euclidean  | normal  |   identity  |   0  |
|  [ghostnet_100_act2_c5_cosine_qe_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_act2_c5_cosine_qe_cccf_224_b256.yaml)   |  ghostnet_100  | 59.128 / 72.913 / 77.321 / 84.063  | act2 |    5    |  identity |   identity  |   cosine  | normal  |   qe  |   0  |
|  [ghostnet_100_fc_c5_cosine_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_fc_c5_cosine_cccf_224_b256.yaml)   |  ghostnet_100  | 56.823 / 73.834 / 80.432 / 87.595  | fc |    5    |  identity |   identity  |   cosine  | normal  |   identity  |   0  |
|  [ghostnet_100_act2_c10_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_act2_c10_cccf_224_b256.yaml)   |  ghostnet_100  | 56.403 / 72.807 / 79.161 / 85.909  | act2 |    10    |  identity |   identity  |   euclidean  | normal  |   identity  |   0  |
|  [ghostnet_100_act2_c5_qe_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_act2_c5_qe_cccf_224_b256.yaml)   |  ghostnet_100  | 55.210 / 68.640 / 75.012 / 83.436  | act2 |    5    |  identity |   identity  |   euclidean  | normal  |   qe  |   0  |
|  [ghostnet_100_fc_c5_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_fc_c5_cccf_224_b256.yaml)   |  ghostnet_100  | 52.987 / 70.589 / 77.698 / 85.729  | fc |    5    |  identity |   identity  |   euclidean  | normal  |   identity  |   0  |
|  [ghostnet_100_act2_c5_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_act2_c5_cccf_224_b256.yaml)   |  ghostnet_100  | 52.887 / 69.544 / 76.302 / 83.906  | act2 |    5    |  identity |   identity  |   euclidean  | normal  |   identity  |   0  |
|  [ghostnet_100_conv_head_c5_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_conv_head_c5_cccf_224_b256.yaml)   |  ghostnet_100  | 52.882 / 69.547 / 76.307 / 83.908  | conv_head |    5    |  identity |   identity  |   euclidean  | normal  |   identity  |   0  |
|  [ghostnet_100_blocks_gem_c5_cosine_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_blocks_gem_c5_cosine_cccf_224_b256.yaml)   |  ghostnet_100  | 47.546 / 64.049 / 70.951 / 78.609  | blocks |    5    |  gem |   identity  |   cosine  | normal  |   identity  |   0  |
|  [ghostnet_100_blocks_gap_c5_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_blocks_gap_c5_cccf_224_b256.yaml)   |  ghostnet_100  | 46.113 / 62.518 / 69.773 / 78.317  | blocks |    5    |  gap |   identity  |   euclidean  | normal  |   identity  |   0  |
|  [ghostnet_100_global_pool_c5_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_global_pool_c5_cccf_224_b256.yaml)   |  ghostnet_100  | 46.110 / 62.513 / 69.773 / 78.317  | global_pool |    5    |  identity |   identity  |   euclidean  | normal  |   identity  |   0  |
|  [ghostnet_100_blocks_crow_c5_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_blocks_crow_c5_cccf_224_b256.yaml)   |  ghostnet_100  | 44.532 / 60.295 / 67.057 / 75.238  | blocks |    5    |  crow |   identity  |   euclidean  | normal  |   identity  |   0  |
|  [ghostnet_100_blocks_gem_c5_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_blocks_gem_c5_cccf_224_b256.yaml)   |  ghostnet_100  | 43.193 / 59.453 / 66.529 / 75.456  | blocks |    5    |  gem |   identity  |   euclidean  | normal  |   identity  |   0  |
|  [ghostnet_100_blocks_gmp_c5_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_blocks_gmp_c5_cccf_224_b256.yaml)   |  ghostnet_100  | 38.350 / 54.525 / 61.709 / 71.012  | blocks |    5    |  gmp |   identity  |   euclidean  | normal  |   identity  |   0  |
|  [ghostnet_100_blocks_spoc_c5_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_blocks_spoc_c5_cccf_224_b256.yaml)   |  ghostnet_100  | 32.148 / 46.092 / 52.478 / 60.970  | blocks |    5    |  spoc |   identity  |   euclidean  | normal  |   identity  |   0  |
|  [ghostnet_100_blocks_r_mac_c5_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_blocks_r_mac_c5_cccf_224_b256.yaml)   |  ghostnet_100  | 25.395 / 38.602 / 45.014 / 53.717  | blocks |    5    |  r_mac |   identity  |   euclidean  | normal  |   identity  |   0  |
|  [ghostnet_100_blocks_c5_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_blocks_c5_cccf_224_b256.yaml)   |  ghostnet_100  | 7.845 / 11.945 / 14.773 / 19.593  | blocks |    5    |  identity |   identity  |   euclidean  | normal  |   identity  |   0  |
