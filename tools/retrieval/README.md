# README

There are three steps for image retrieval, that is

1. Feature extraction
2. Feature retrieval
3. Retrieval evaluation

## Feature Extraction

First, you should appoint `model_arch` and feature `layer`, then you should input image paths. we provide `General`
dataset class, it needs two files:

1. data.csv: each line represents an image path and its corresponding label in the following format:
    ```text
    /path/to/img1.jpg,,label1
    /path/to/img２.jpg,,label2
    ... 
    ```
2. cls.csv: each line represents a class name, and it corresponds to the label:
    ```text
    cls1
    cls2
    ...
    ```

Then, execute the program according to the following code

```shell
python extract_features.py --model-arch resnet50 --layer fc --dataset General --image-dir data/train/ --save-dir data/gallery_fc
python extract_features.py --model-arch resnet50 --layer fc --dataset General --image-dir data/test/ --save-dir data/query_fc
```

Finally, you can find features in `save-dir`. The features of each image are saved according to the image name, and you
can find `info.pkl`, it includes the following contents:

```text
'feat': layer,
'model': model_arch,
'pretrained': pretrained,
'classes': classes,
'content': {'img_name1': label, 'img_name2': label, ...}
```

## Feature Retrieval

Set the query set `query-dir` and the gallery set `gallery-dir` to perform feature retrieval

```shell
python retrieval_features.py --query-dir data/query_fc --gallery-dir data/gallery_fc --save-dir data/retrieval_fc
python retrieval_features.py --query-dir data/query_fc --gallery-dir data/gallery_fc --save-dir data/retrieval_fc --topk 20
```

In the `save-dir`, you can find the sorting results of each query feature retrieval. There is also a file `info.pkl`, it
includes the following contents:

```text
'classes': query_cls_list,
'content': {query_name1: label1, query_name2: label2, ...},
'query_dir': query_dir,
'gallery_dir': gallery_dir
```

## Retrieval Evaluation

Given the retrieval result path, read `info.pkl` file, read the retrieval results in turn. Evaluate retrieval
performance, such as `ACCURACY/PRECISION/MAP`:

```shell
python evaluate_features.py --retrieval-dir data/retrieval_fc --retrieval-type ACCURACY
```

## Examples

### Oxford5k(Old)

```shell
# Set env
export PYTHONPATH=/path/to/SimpleIR/
python tools/data/make_oxford5k_paris6k.py
# Extract gallery feature
python tools/retrieval/extract_features.py \
 --model-arch resnet50 \
 --layer avgpool \
 --gallery \
 --dataset Oxford \
 --image-dir data/oxford \
 --save-dir feature/oxford/gallery \
 --aggregate IDENTITY \
 --enhance IDENTITY
# Extract query feature
python tools/retrieval/extract_features.py \
 --model-arch resnet50 \
 --layer avgpool \
 --dataset Oxford \
 --image-dir data/oxford \
 --save-dir feature/oxford/query \
 --aggregate IDENTITY \
 --enhance IDENTITY
# Retrieval feature
python tools/retrieval/retrieval_features.py \
 --query-dir feature/oxford/query \
 --gallery-dir feature/oxford/gallery \
 --distance EUCLIDEAN \
 --rank NORMAL \
 --rerank IDENTITY \
 --save-dir retrieval/oxford
# Evaluate mAP_for_Oxford
python tools/retrieval/evaluate_features.py \
 --retrieval-dir retrieval/oxford \
 --retrieval-type MAP_OXFORD \
 --query-dir data/oxford
...
...
[11/01 21:15:19][INFO] evaluate_features.py:  46: => MAP: 43.298%
```

You can also use config file

```shell
bash tools/bash_eval.sh tools/configs/resnet50_avgpool_oxford_224_b32.yaml
...
...
[11/01 21:37:25][INFO] infer.py:  61:  * => MAP 43.296% 
```

### Oxford5k(new)

```shell
# Set env
export PYTHONPATH=/path/to/SimpleIR/
python tools/data/make_roxford5k_rparis6k.py
# Extract gallery feature
python tools/retrieval/extract_features.py \
 --model-arch resnet50 \
 --layer avgpool \
 --gallery \
 --dataset Oxford5k \
 --image-dir data/oxford5k \
 --save-dir feature/oxford5k/gallery \
 --aggregate IDENTITY \
 --enhance IDENTITY
# Extract query feature
python tools/retrieval/extract_features.py \
 --model-arch resnet50 \
 --layer avgpool \
 --dataset Oxford5k \
 --image-dir data/oxford5k \
 --save-dir feature/oxford5k/query \
 --aggregate IDENTITY \
 --enhance IDENTITY
# Retrieval feature
python tools/retrieval/retrieval_features.py \
 --query-dir feature/oxford5k/query \
 --gallery-dir feature/oxford5k/gallery \
 --distance EUCLIDEAN \
 --rank NORMAL \
 --rerank IDENTITY \
 --save-dir retrieval/oxford5k
# Evaluate mAP_for_ROxford
python tools/retrieval/evaluate_features.py \
  --retrieval-dir retrieval/oxford5k \
  --retrieval-type  MAP_ROXFORD \
  --query-dir data/oxford5k \
  --dataset oxford5k
...
...
[11/02 13:58:14][INFO] evaluate_features.py:  56: >> oxford5k: mAP 46.27
```

You can also use config file

```shell
bash tools/bash_eval.sh tools/configs/resnet50_avgpool_oxford_224_b32_new.yaml
...
...
[11/02 13:55:49][INFO] infer.py:  59: >> oxford5k: mAP 46.27
```

### ROxford5k(new)

```shell
# Set env
export PYTHONPATH=/path/to/SimpleIR/
python tools/data/make_roxford5k_rparis6k.py --dataset  roxford5k --root ./data/roxford5k
# Use config file
bash tools/bash_eval.sh tools/configs/resnet50_avgpool_roxford_224_b32.yaml
...
...
[11/02 14:04:13][INFO] infer.py:  67: >> roxford5k: mAP E: 38.67, M: 26.86, H: 5.94
[11/02 14:04:13][INFO] infer.py:  70: >> roxford5k: mP@k(1, 5, 10) E: [64.71 51.37 46.16], M: [64.29 50.86 45.29], H: [20.   15.71 12.92]
```