# Get started

## Create Dataset

Create toy dataset

```shell
python tools/dataset/extract_torchvision_dataset.py --dataset CIFAR100 --toy ../datasets/cifar100
```

For custom datasets, set the format as follows:

```text
root/
    train/
        cate_1/
            xxx.jpg
        cate_2/
            xxx.jpg
    val/
        cate_1/
            xxx.jpg
        cate_2/
            xxx.jpg
```

## Extract Features

```shell
python extract.py --arch resnet18 --data toy.yaml
```

After completing the extraction, you can view the file in `runs/extract/expXXX/`

## Retrieval Features

```shell
python retrieval.py /path/to/gallery.pkl /path/to/query.pkl
```

After completing the retrieval, the retrieval results will be saved in `runs/retrieval/expXXX/info.pkl`

## Metric Features

```shell
python metric.py /path/to/retrieval.pkl --evaluate ACCURACY
```