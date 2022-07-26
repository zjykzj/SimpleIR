
# Install

## Requirements


    Linux (Windows is not officially supported)
    Python 3.8
    PyTorch 1.8.2 or higher
    torchvison 0.9.2 or higher

About `apex`，there has a bug in nvidia/apex, see [How can I solve it？ #1215](https://github.com/NVIDIA/apex/issues/1215). Now i use pytorch naive apex util in ZCls2.

I recommend people to use docker training environment, see

* [NGC Pytorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
* [PyTorch Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)
* [NVIDIA/apex](https://github.com/NVIDIA/apex)

## Install SimpleIR

### Source install

* Clone the `SimpleIR` repository.

```
git clone https://github.com/zjykzj/SimpleIR.git
cd ZCls2
```

* Install `SimpleIR`.

```
python3 setup.py install
```

### Pip install (*RECOMMEND*)

```python
pip3 install simpleir
```