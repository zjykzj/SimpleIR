
# CHANGE

## v0.3.0

* New features
  1. feat(model): add GhostNet. [805530e9c](https://github.com/zjykzj/SimpleIR/tree/805530e9c216e41cf612d4a659fb059444b669e1)
* Bug fixes
* Breaking changes
  1. refactor(simpleir): refactor criterion use. [e71294a](https://github.com/zjykzj/SimpleIR/tree/e71294a9e7423b7167436837e097561942349b4d)

## v0.2.1

* New features
  1. feat(python): add setup.py. [1c1b6c59](https://github.com/zjykzj/SimpleIR/tree/1c1b6c59d9e0b1deb217c42e8fbf2223e8d837a3)
  2. fix(trainer.py): fix wrong use param i. [2f929567](https://github.com/zjykzj/SimpleIR/tree/2f92956761d9905b04e5319678b56caa3e098b9f)
* Bug fixes
* Breaking changes.

## v0.2.0 ([1c5b581e](https://github.com/zjykzj/SimpleIR/tree/1c5b581e3d96c76364472dcce8448561288611c0))

* New features
  1. build(python): update requirements.txt. [93351fe1](https://github.com/zjykzj/SimpleIR/tree/93351fe1111a37b29909621de0ed5b012d592918)
  2. feat(metric): before compute similarity, add Feat Enhance step. [31d4db33](https://github.com/zjykzj/SimpleIR/tree/31d4db3324d5846185265e93746d3c190e4db4bd)
  3. feat(simpleir): add metric module. [86b73969](https://github.com/zjykzj/SimpleIR/tree/86b73969db156d2e150535edd9d4b2e19c11bbe1)
     1. add Similarity module.
     2. add Rank module.
  4. perf(simpleir): add cfg.TRAIN.TOP_K support. [cc7bf3f3](https://github.com/zjykzj/SimpleIR/tree/cc7bf3f3c073bdb480a5c78841d0b07c20e4f772)
* Bug fixes
* Breaking changes.

## v0.1.0 ([c6b751f](https://github.com/zjykzj/SimpleIR/commit/c6b751f56aeb977d0fdb9720eaa6f04441910abe))

* New features 
  1. feat(simpleir): refactor SimpleIR training module. [96b9ffb](https://github.com/zjykzj/SimpleIR/commit/96b9ffbd019587a340149956ccf5fa891d928d66)
     1. refactor trainer/infer module;
     2. add resnet/TinyAutocoder support;
     3. add MSELoss support;
     4. modify val_loader definition.
  2. Make query and gallery for Caltech101. [8b8bdc1](https://github.com/zjykzj/SimpleIR/commit/8b8bdc1034ac5e4317583d7cfe6a2133dac20f80)
  3. Make the first auto-encoder model training. [ff00a8f](https://github.com/zjykzj/SimpleIR/commit/ff00a8ff9fc26a91b81bfb91b8bef64f752f0c80)
* Bug fixes
* Breaking changes.
