
# LeaderBoard (Based on CCCF)

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
|  [ghostnet_100_act2_c5_cccf_224_b256_e90_g4](../configs/cccf/ghostnet/ghostnet_100_act2_c5_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 48.191 / 63.768 / 70.201 / 77.878  | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |
|  [r50_avgpool_c5_cccf_224_b256_e90_g4](../configs/cccf/resnet/r50_avgpool_c5_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 44.413 / 61.038 / 67.892 / 76.531  | CrossEntropyLoss |    SGD    |  MultiStepLR |   90  |   True  |

### SCORES (Eval)

| cfg |    model   |   top1/top3/top5/top10   |   feat_type   | max_num | aggregate | enhance | distance | rank |
|:---:|:----------:|:-------------:|:----------------:|:---------:|:------------:|:-----:|:-----:|:-----:|
|  [ghostnet_100_act2_c50_cosine_cccf_224_b256_e90_g4](configs/cccf/ghostnet/ghostnet_100_act2_c50_cosine_cccf_224_b256_e90_g4.yaml)   |  ghostnet_100  | 58.102 / 72.389 / 77.354 / 83.366   | act2 |    50    |  identity |   identity  |   cosine  | normal  |
|  [mobilenet_v3_large_hardswish_c40_cosine_cccf_224_b256_e90_g4](../configs/cccf/mobilenet/mobilenet_v3_large_hardswish_c40_cosine_cccf_224_b256_e90_g4.yaml)   |  mobilenet_v3_large  | 57.204 / 71.445 / 76.924 / 82.740   | hardswish |    40    |  identity |   identity  |   cosine  | normal  |
|  [r50_layer4_crow_c50_cosine_cccf_224_b256_e90_g4](configs/cccf/resnet/r50_layer4_crow_c50_cosine_cccf_224_b256_e90_g4.yaml)   |  resnet50  | 53.950 / 68.322 / 74.381 / 81.085 | layer4 |    50    |  crow |   identity  |   cosine  | normal  |
