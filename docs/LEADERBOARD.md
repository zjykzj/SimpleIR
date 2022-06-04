
# LeaderBoard (Based on CCCF)

## About CCCF

    CCCF is a custom mixed classification dataset, including

    1. CIFAR100: https://paperswithcode.com/dataset/cifar-100
    2. CUB-200-2011: https://paperswithcode.com/dataset/cub-200-2011
    3. Caltech-101: https://paperswithcode.com/dataset/caltech-101
    4. Food-101: https://paperswithcode.com/dataset/food-101

    The classes num = 100 + 200 + 101 + 101 = 502

## SCORES (Eval)

| cfg |    model   |   top1/top3/top5/top10   |   feat_type   | max_num | aggregate | enhance | distance | rank | re_rank | index_mode |
|:---:|:----------:|:-------------:|:----------------:|:---------:|:------------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|  [ghostnet_100_act2_c50_cosine_cccf_224_b256](../configs/cccf/ghostnet/ghostnet_100_act2_c50_cosine_cccf_224_b256.yaml)   |  ghostnet_100  | 66.809 / 80.152 / 84.811 / 89.895   | act2 |    50    |  identity |   identity  |   cosine  | normal  |   identity  |   0  |
|  [mobilenet_v3_large_hardswish_c40_cosine_cccf_224_b256_e90_g4](configs/cccf/mobilenet/mobilenet_v3_large_hardswish_c40_cosine_cccf_224_b256_e90_g4.yaml)   |  mobilenet_v3_large  | 57.204 / 71.445 / 76.924 / 82.740   | hardswish |    40    |  identity |   identity  |   cosine  | normal  |   identity  |   0  |
|  [r50_layer4_crow_c50_cosine_cccf_224_b256_e90_g4](configs/cccf/resnet/r50_layer4_crow_c50_cosine_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 53.950 / 68.322 / 74.381 / 81.085 | layer4 |    50    |  crow |   identity  |   cosine  | normal  |   identity  |   0  |
