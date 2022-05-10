
# README

## Train

[05/09 18:37:40][INFO] train.py: 175:  BestEpoch: [81]
[05/09 18:37:40][INFO] train.py: 179:  * Prec@1 44.413 Prec@3 61.038 Prec@5 67.892 Prec@10 76.531 
[05/09 18:37:40][INFO] checkpoint.py:  21: Save to outputs/r50_avgpool_cccf_224_b256_e90_g4/checkpoint_90.pth.tar

[05/09 18:17:49][INFO] train.py: 175:  BestEpoch: [90]
[05/09 18:17:49][INFO] train.py: 179:  * Prec@1 48.191 Prec@3 63.768 Prec@5 70.201 Prec@10 77.878 
[05/09 18:17:49][INFO] checkpoint.py:  21: Save to outputs/ghostnet_100_act2_cccf_224_b256_e90_g4/checkpoint_90.pth.tar
[05/09 18:17:49][INFO] checkpoint.py:  25: Copy to outputs/ghostnet_100_act2_cccf_224_b256_e90_g4/model_best.pth.tar

## Eval

### GhostNet

#### cate_num 5

act2 cate_num 5
[05/10 02:24:27][INFO] infer.py:  96:  * Prec@1 48.172 Prec@3 63.740 Prec@5 70.201 Prec@10 77.878 

act2 cate_num 5 cosine
[05/10 06:11:55][INFO] infer.py:  96:  * Prec@1 54.904 Prec@3 70.341 Prec@5 75.961 Prec@10 83.226 

blocks
[05/10 02:28:30][INFO] infer.py:  96:  * Prec@1 6.255 Prec@3 9.696 Prec@5 11.650 Prec@10 16.568

blocks crow
[05/10 02:28:42][INFO] infer.py:  96:  * Prec@1 39.785 Prec@3 54.867 Prec@5 61.169 Prec@10 69.799 

blocks gap
[05/10 02:47:44][INFO] infer.py:  96:  * Prec@1 41.206 Prec@3 56.802 Prec@5 63.806 Prec@10 72.099

blocks gem
[05/10 02:47:53][INFO] infer.py:  96:  * Prec@1 38.308 Prec@3 54.053 Prec@5 61.178 Prec@10 70.173

blocks gem cosine
[05/10 06:22:19][INFO] infer.py:  96:  * Prec@1 43.385 Prec@3 59.645 Prec@5 66.227 Prec@10 73.922

blocks gmp
[05/10 02:49:01][INFO] infer.py:  96:  * Prec@1 33.857 Prec@3 49.687 Prec@5 56.980 Prec@10 66.339

blocks r_mac
[05/10 02:49:31][INFO] infer.py:  96:  * Prec@1 23.114 Prec@3 35.718 Prec@5 42.338 Prec@10 51.304

blocks spoc
[05/10 02:50:29][INFO] infer.py:  96:  * Prec@1 28.957 Prec@3 41.861 Prec@5 48.153 Prec@10 56.260

conv_head
[05/10 02:50:58][INFO] infer.py:  96:  * Prec@1 48.172 Prec@3 63.759 Prec@5 70.201 Prec@10 77.887 

conv_head cosine
[05/10 06:21:35][INFO] infer.py:  96:  * Prec@1 54.904 Prec@3 70.341 Prec@5 75.970 Prec@10 83.207

fc
[05/10 02:51:52][INFO] infer.py:  96:  * Prec@1 47.564 Prec@3 64.778 Prec@5 71.763 Prec@10 80.112

fc cosine
[05/10 06:16:08][INFO] infer.py:  96:  * Prec@1 51.697 Prec@3 67.742 Prec@5 74.596 Prec@10 81.917

global_pool
[05/10 02:52:33][INFO] infer.py:  96:  * Prec@1 41.197 Prec@3 56.812 Prec@5 63.824 Prec@10 72.137

#### cate_num 10

act2 cate_num 10
[05/10 02:57:31][INFO] infer.py:  96:  * Prec@1 50.229 Prec@3 65.732 Prec@5 72.052 Prec@10 79.243

#### cate_num 20

act2 cate_num 20
[05/10 02:57:56][INFO] infer.py:  96:  * Prec@1 51.360 Prec@3 66.919 Prec@5 72.894 Prec@10 79.364 

#### cate_num 30

act2 cate_num 30
[05/10 02:59:29][INFO] infer.py:  96:  * Prec@1 51.903 Prec@3 67.181 Prec@5 73.249 Prec@10 79.645 

#### cate_num 40

act2 cate_num 40
[05/10 02:59:37][INFO] infer.py:  96:  * Prec@1 52.258 Prec@3 67.377 Prec@5 73.268 Prec@10 79.738

#### cate_num 50

act2 cate_num 50
[05/10 03:01:45][INFO] infer.py:  96:  * Prec@1 52.445 Prec@3 67.377 Prec@5 73.277 Prec@10 79.785

act2 cate_num 50 cosine
[05/10 06:34:36][INFO] infer.py:  96:  * Prec@1 58.102 Prec@3 72.389 Prec@5 77.354 Prec@10 83.366

conv_head
[05/10 03:05:14][INFO] infer.py:  96:  * Prec@1 52.445 Prec@3 67.377 Prec@5 73.277 Prec@10 79.776 

conv_head cosine
[05/10 06:35:25][INFO] infer.py:  96:  * Prec@1 58.093 Prec@3 72.380 Prec@5 77.373 Prec@10 83.375

fc
[05/10 03:06:17][INFO] infer.py:  96:  * Prec@1 51.716 Prec@3 67.546 Prec@5 73.932 Prec@10 80.888 

fc cosine
[05/10 06:40:35][INFO] infer.py:  96:  * Prec@1 55.390 Prec@3 70.407 Prec@5 76.185 Prec@10 82.422 

#### cate_num 60

act2 cate_num 60
[05/10 03:01:51][INFO] infer.py:  96:  * Prec@1 52.436 Prec@3 67.415 Prec@5 73.324 Prec@10 79.719

#### cate_num 70

act2 cate_num 70
[05/10 03:04:08][INFO] infer.py:  96:  * Prec@1 52.417 Prec@3 67.415 Prec@5 73.305 Prec@10 79.701

### ResNet

#### cate_num 5

avgpool cate_num 5
[05/10 01:51:48][INFO] infer.py:  96:  * Prec@1 44.413 Prec@3 61.038 Prec@5 67.892 Prec@10 76.531

avgpool cate_num 5 cosine
[05/10 06:58:20][INFO] infer.py:  96:  * Prec@1 48.883 Prec@3 64.741 Prec@5 71.725 Prec@10 79.448 

fc
[05/10 01:53:31][INFO] infer.py:  96:  * Prec@1 41.870 Prec@3 58.775 Prec@5 66.321 Prec@10 75.475

layer4
[05/10 01:56:23][INFO] infer.py:  96:  * Prec@1 23.413 Prec@3 35.138 Prec@5 40.954 Prec@10 49.444

layer4 crow
[05/10 01:57:18][INFO] infer.py:  96:  * Prec@1 45.573 Prec@3 61.524 Prec@5 68.537 Prec@10 76.774

layer4 crow cosine
[05/10 06:58:08][INFO] infer.py:  96:  * Prec@1 49.472 Prec@3 65.507 Prec@5 72.071 Prec@10 79.598

layer4 gap
[05/10 01:58:19][INFO] infer.py:  96:  * Prec@1 44.413 Prec@3 61.038 Prec@5 67.892 Prec@10 76.540

layer4 gem
[05/10 01:59:17][INFO] infer.py:  96:  * Prec@1 44.320 Prec@3 60.701 Prec@5 67.770 Prec@10 76.064

layer4 gmp
[05/10 02:01:10][INFO] infer.py:  96:  * Prec@1 41.356 Prec@3 58.065 Prec@5 65.199 Prec@10 74.296

layer4 r_mac
[05/10 02:01:59][INFO] infer.py:  96:  * Prec@1 26.526 Prec@3 42.057 Prec@5 49.331 Prec@10 60.056 

layer4 spoc
[05/10 02:02:49][INFO] infer.py:  96:  * Prec@1 36.765 Prec@3 52.062 Prec@5 58.943 Prec@10 68.060 

#### cate_num 10

avgpool cate_num 10
[05/10 02:04:24][INFO] infer.py:  96:  * Prec@1 46.489 Prec@3 62.880 Prec@5 69.547 Prec@10 77.962

#### cate_num 20

avgpool cate_num 20
[05/10 02:05:27][INFO] infer.py:  96:  * Prec@1 48.237 Prec@3 64.404 Prec@5 70.771 Prec@10 78.635

avgpool cate_num 20 cosine
[05/10 08:00:13][INFO] infer.py:  96:  * Prec@1 52.426 Prec@3 67.779 Prec@5 73.726 Prec@10 81.103

layer4 crow
[05/10 02:15:21][INFO] infer.py:  96:  * Prec@1 49.294 Prec@3 64.909 Prec@5 71.248 Prec@10 78.373 

layer4 crow cosine
[05/10 07:56:36][INFO] infer.py:  96:  * Prec@1 53.343 Prec@3 68.079 Prec@5 74.240 Prec@10 81.197

layer4 gap
[05/10 02:18:01][INFO] infer.py:  96:  * Prec@1 48.256 Prec@3 64.404 Prec@5 70.771 Prec@10 78.644

layer4 gem
[05/10 02:19:46][INFO] infer.py:  96:  * Prec@1 48.602 Prec@3 64.329 Prec@5 70.631 Prec@10 78.074

#### cate_num 30

avgpool cate_num 30
[05/10 02:06:57][INFO] infer.py:  96:  * Prec@1 48.658 Prec@3 64.815 Prec@5 70.827 Prec@10 78.906 

#### cate_num 40

avgpool cate_num 40
[05/10 02:07:46][INFO] infer.py:  96:  * Prec@1 48.911 Prec@3 65.030 Prec@5 71.136 Prec@10 79.046

#### cate_num 50

avgpool cate_num 50
[05/10 02:09:29][INFO] infer.py:  96:  * Prec@1 49.079 Prec@3 65.115 Prec@5 71.248 Prec@10 79.074 

avgpool cate_num 50 cosine
[05/10 08:17:58][INFO] infer.py:  96:  * Prec@1 52.959 Prec@3 68.228 Prec@5 74.128 Prec@10 81.169

layer4 crow
[05/10 02:16:00][INFO] infer.py:  96:  * Prec@1 50.098 Prec@3 65.395 Prec@5 71.678 Prec@10 78.700 

layer4 crow cosine
[05/10 08:14:28][INFO] infer.py:  96:  * Prec@1 53.950 Prec@3 68.322 Prec@5 74.381 Prec@10 81.085

layer4 gap
[05/10 02:18:14][INFO] infer.py:  96:  * Prec@1 49.079 Prec@3 65.115 Prec@5 71.258 Prec@10 79.084

layer4 gem
[05/10 02:20:16][INFO] infer.py:  96:  * Prec@1 49.397 Prec@3 64.918 Prec@5 71.080 Prec@10 78.298 

#### cate_num 60

avgpool cate_num 60
[05/10 02:12:06][INFO] infer.py:  96:  * Prec@1 49.023 Prec@3 65.068 Prec@5 71.258 Prec@10 79.093

#### cate_num 70

avgpool cate_num 70
[05/10 02:13:15][INFO] infer.py:  96:  * Prec@1 49.032 Prec@3 65.049 Prec@5 71.248 Prec@10 79.112