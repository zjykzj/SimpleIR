
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

[comment]: <> (| cfg |    model   |   top1/top3/top5/top10   |   feat_type   | max_num | aggregate | enhance | distance | rank |)

[comment]: <> (|:---:|:----------:|:-------------:|:----------------:|:---------:|:------------:|:-----:|:-----:|:-----:|)

[comment]: <> (|  [mobilenet_v3_large_hardswish_c40_cosine_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_hardswish_c40_cosine_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 57.204 / 71.445 / 76.924 / 82.740   | hardswish |    40    |  identity |   identity  |   cosine  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_hardswish_c30_cosine_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_hardswish_c30_cosine_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 57.092 / 71.323 / 76.868 / 82.730   | hardswish |    30    |  identity |   identity  |   cosine  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_hardswish_c20_cosine_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_hardswish_c20_cosine_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 56.858 / 71.220 / 76.662 / 82.805   | hardswish |    20    |  identity |   identity  |   cosine  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_linear_c20_cosine_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_linear_c20_cosine_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 56.858 / 71.220 / 76.662 / 82.805   | linear |    20    |  identity |   identity  |   cosine  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_hardswish_c10_cosine_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_hardswish_c10_cosine_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 55.203 / 70.351 / 76.213 / 82.618   | hardswish |    10    |  identity |   identity  |   cosine  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_linear_c10_cosine_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_linear_c10_cosine_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 55.157 / 70.341 / 76.204 / 82.618   | linear |    10    |  identity |   identity  |   cosine  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_hardswish_c5_cosine_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_hardswish_c5_cosine_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 52.894 / 69.060 / 75.175 / 82.347   | hardswish |    5    |  identity |   identity  |   cosine  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_linear_c5_cosine_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_linear_c5_cosine_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 52.885 / 69.070 / 75.175 / 82.347   | linear |    5    |  identity |   identity  |   cosine  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_avgpool_c10_cosine_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_avgpool_c10_cosine_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 49.257 / 64.628 / 70.856 / 78.149   | avgpool |    10    |  identity |   identity  |   cosine  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_avgpool_c5_cosine_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_avgpool_c5_cosine_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 49.247 / 64.656 / 70.856 / 78.149   | avgpool |    5    |  identity |   identity  |   cosine  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_classifier_c5_cosine_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_classifier_c5_cosine_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 48.396 / 64.899 / 72.118 / 79.710   | classifier |    5    |  identity |   identity  |   cosine  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_classifier_c5_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_classifier_c5_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 44.647 / 61.692 / 69.201 / 77.803   | classifier |    5    |  identity |   identity  |   euclidean  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_linear_c5_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_linear_c5_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 44.180 / 60.281 / 67.396 / 75.559   | linear |    5    |  identity |   identity  |   euclidean  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_hardswish_c5_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_hardswish_c5_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 44.161 / 60.281 / 67.405 / 75.540   | hardswish |    5    |  identity |   identity  |   euclidean  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_blocks_gap_c5_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_blocks_gap_c5_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 42.590 / 58.560 / 65.554 / 73.427   | blocks |    5    |  gap |   identity  |   euclidean  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_avgpool_c5_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_avgpool_c5_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 42.581 / 58.541 / 65.545 / 73.427   | avgpool |    5    |  identity |   identity  |   euclidean  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_blocks_gmp_c5_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_blocks_gmp_c5_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 37.831 / 53.539 / 60.346 / 68.920   | blocks |    5    |  gmp |   identity  |   euclidean  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_blocks_spoc_c5_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_blocks_spoc_c5_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 33.717 / 47.115 / 53.661 / 61.805   | blocks |    5    |  spoc |   identity  |   euclidean  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_blocks_r_mac_c5_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_blocks_r_mac_c5_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 28.331 / 42.712 / 49.799 / 58.597   | blocks |    5    |  r_mac |   identity  |   euclidean  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_blocks_c5_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_blocks_c5_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 15.035 / 21.337 / 24.507 / 29.677   | blocks |    5    |  identity |   identity  |   euclidean  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_blocks_crow_c5_cosine_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_blocks_crow_c5_cosine_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  |  0.776 / 1.262 / 1.440 / 1.711  | blocks |    5    |  crow |   identity  |   cosine  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_blocks_crow_c5_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_blocks_crow_c5_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  |  0.664 / 1.169 / 1.356 / 1.674  | blocks |    5    |  crow |   identity  |   euclidean  | normal  |)

[comment]: <> (|  [mobilenet_v3_large_blocks_gem_c5_cccf_224_b256_e90_g4]&#40;../configs/cccf/mobilenet/mobilenet_v3_large_blocks_gem_c5_cccf_224_b256_e90_g4.yaml&#41;   |  mobilenet_v3_large  | 0.168 / 0.486 / 0.589 / 0.823   | blocks |    5    |  gem |   identity  |   euclidean  | normal  |)






