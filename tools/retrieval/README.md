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