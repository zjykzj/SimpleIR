
# DEMO

>Demo for how to use SimpleIR projects.

## Prepare

```shell
cd demo/dataset/
# extract MNIST
bash run-mnist.sh
# extract CIFAR10
bash run-cifar10.sh
```

The data can be found in `./data/mnist` and `./data/cifar10`

## Classify Model

How to Use Classify Model to Realize Image Retrieval?

```shell
cd demo/resnet/
bash train.sh
```

## AutoCoder Model

How to Use AutoCoder Model to Realize Image Retrieval?

```shell
cd demo/autocoder/
bash train.sh
```