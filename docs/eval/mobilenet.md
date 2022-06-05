
# MobileNet

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
|  [mobilenet_v3_large_hardswish_c5_cccf_224_b256_e90_g4](../../configs/cccf/mobilenet/mobilenet_v3_large_hardswish_c5_cccf_224_b256_e90_g4.yaml)   |  mobilenet_v3_large  | 50.984 / 67.431 / 74.042 / 81.917  | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |

### SCORES (Eval)

| cfg |    model   |   top1/top3/top5/top10   |   feat_type   | max_num | aggregate | enhance | distance | rank |
|:---:|:----------:|:-------------:|:----------------:|:---------:|:------------:|:-----:|:-----:|:-----:|
|  [mobilenet_v3_large_hardswish_c50_cosine_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_hardswish_c50_cosine_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 66.748 / 80.033 / 84.808 / 89.963   | hardswish |    50    |  identity |   identity  |   cosine  | normal  |
|  [mobilenet_v3_large_hardswish_c40_cosine_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_hardswish_c40_cosine_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 66.414 / 79.888 / 84.687 / 89.918   | hardswish |    40    |  identity |   identity  |   cosine  | normal  |
|  [mobilenet_v3_large_hardswish_c30_cosine_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_hardswish_c30_cosine_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 65.837 / 79.584 / 84.486 / 89.853   | hardswish |    30    |  identity |   identity  |   cosine  | normal  |
|  [mobilenet_v3_large_hardswish_c20_cosine_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_hardswish_c20_cosine_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 64.974 / 78.855 / 84.123 / 89.614   | hardswish |    20    |  identity |   identity  |   cosine  | normal  |
|  [mobilenet_v3_large_linear_c20_cosine_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_linear_c20_cosine_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 64.965 / 78.866 / 84.126 / 89.607   | linear |    20    |  identity |   identity  |   cosine  | normal  |
|  [mobilenet_v3_large_hardswish_c10_cosine_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_hardswish_c10_cosine_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 62.515 / 77.567 / 83.116 / 89.226   | hardswish |    10    |  identity |   identity  |   cosine  | normal  |
|  [mobilenet_v3_large_linear_c10_cosine_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_linear_c10_cosine_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 62.513 / 77.567 / 83.111 / 89.233   | linear |    10    |  identity |   identity  |   cosine  | normal  |
|  [mobilenet_v3_large_hardswish_c5_cosine_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_hardswish_c5_cosine_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 59.203 / 75.311 / 81.540 / 88.219   | hardswish |    5    |  identity |   identity  |   cosine  | normal  |
|  [mobilenet_v3_large_linear_c5_cosine_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_linear_c5_cosine_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 59.212 / 75.309 / 81.540 / 88.223   | linear |    5    |  identity |   identity  |   cosine  | normal  |
|  [mobilenet_v3_large_avgpool_c10_cosine_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_avgpool_c10_cosine_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 59.119 / 74.165 / 79.635 / 85.986   | avgpool |    10    |  identity |   identity  |   cosine  | normal  |
|  [mobilenet_v3_large_avgpool_c5_cosine_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_avgpool_c5_cosine_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 55.028 / 70.493 / 76.830 / 83.427   | avgpool |    5    |  identity |   identity  |   cosine  | normal  |
|  [mobilenet_v3_large_classifier_c5_cosine_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_classifier_c5_cosine_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 54.266 / 71.793 / 78.784 / 86.641   | classifier |    5    |  identity |   identity  |   cosine  | normal  |
|  [mobilenet_v3_large_linear_c5_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_linear_c5_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 50.989 / 67.431 / 74.032 / 81.912   | linear |    5    |  identity |   identity  |   euclidean  | normal  |
|  [mobilenet_v3_large_avgpool_c5_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_avgpool_c5_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 50.986 / 67.433 / 74.037 / 81.903   | avgpool |    5    |  identity |   identity  |   euclidean  | normal  |
|  [mobilenet_v3_large_hardswish_c5_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_hardswish_c5_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 50.982 / 67.419 / 74.039 / 81.898   | hardswish |    5    |  identity |   identity  |   euclidean  | normal  |
|  [mobilenet_v3_large_classifier_c5_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_classifier_c5_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 50.690 / 68.619 / 75.788 / 84.205   | classifier |    5    |  identity |   identity  |   euclidean  | normal  |
|  [mobilenet_v3_large_blocks_gap_c5_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_blocks_gap_c5_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 48.955 / 65.533 / 72.445 / 80.316   | blocks |    5    |  gap |   identity  |   euclidean  | normal  |
|  [mobilenet_v3_large_blocks_gmp_c5_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_blocks_gmp_c5_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 43.184 / 59.694 / 66.622 / 75.187   | blocks |    5    |  gmp |   identity  |   euclidean  | normal  |
|  [mobilenet_v3_large_blocks_spoc_c5_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_blocks_spoc_c5_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 37.887 / 52.892 / 59.350 / 67.384   | blocks |    5    |  spoc |   identity  |   euclidean  | normal  |
|  [mobilenet_v3_large_blocks_r_mac_c5_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_blocks_r_mac_c5_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 31.863 / 47.223 / 54.065 / 63.434   | blocks |    5    |  r_mac |   identity  |   euclidean  | normal  |
|  [mobilenet_v3_large_blocks_c5_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_blocks_c5_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 17.782 / 24.848 / 28.315 / 33.756   | blocks |    5    |  identity |   identity  |   euclidean  | normal  |
|  [mobilenet_v3_large_blocks_crow_c5_cosine_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_blocks_crow_c5_cosine_cccf_224_b256.yaml)   |  mobilenet_v3_large  |  1.164 / 1.456 / 1.592 / 1.784  | blocks |    5    |  crow |   identity  |   cosine  | normal  |
|  [mobilenet_v3_large_blocks_crow_c5_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_blocks_crow_c5_cccf_224_b256.yaml)   |  mobilenet_v3_large  |  1.101 / 1.400 / 1.533 / 1.739  | blocks |    5    |  crow |   identity  |   euclidean  | normal  |
|  [mobilenet_v3_large_blocks_gem_c5_cccf_224_b256](../configs/cccf/mobilenet/mobilenet_v3_large_blocks_gem_c5_cccf_224_b256.yaml)   |  mobilenet_v3_large  | 0.580 / 0.715 / 0.778 / 0.912   | blocks |    5    |  gem |   identity  |   euclidean  | normal  |
