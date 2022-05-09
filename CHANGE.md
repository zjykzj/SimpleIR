
# CHANGE

## v0.5.0

* New features
  1. perf(rank): make batch rank process. [2b5f5dc7](https://github.com/zjykzj/SimpleIR/tree/2b5f5dc73e36df1b48a6d3f5dd8a20b061ef2e6f)
  2. feat(aggregator): add CroW. [ff90d4ee](https://github.com/zjykzj/SimpleIR/tree/ff90d4ee760e7129d63aa41ab965a5d1b3e91f06)
  3. feat(aggregator): add SpoC. [4c9a5535](https://github.com/zjykzj/SimpleIR/tree/4c9a55356aede97ed067f1e5a7cb09aee242b6d1)
  4. feat(aggregator): add R_Mac. [799f08da](https://github.com/zjykzj/SimpleIR/tree/799f08da43c779d1c12df48b656c6053400bf671)
  5. feat(aggregator): add GeM. [1a90fabc](https://github.com/zjykzj/SimpleIR/tree/1a90fabcbaba947c214c54354e40f3f5791398f6)
  6. feat(metric): add Aggregator for Conv feats. [87b1838](https://github.com/zjykzj/SimpleIR/tree/87b183812f5fcd33bfd700a41efe59a221641931)
  7. feat(config): add cfg.METRIC.MAX_CATE_NUM. [1ea469d](https://github.com/zjykzj/SimpleIR/tree/1ea469d15174936346101191ac5d34ed108b22ab)
  8. perf(similarity): speed up calcluation. [c2679b91](https://github.com/zjykzj/SimpleIR/tree/c2679b91b1bcc427027ff8504ebe48f7606df19d)
  9. perf(metric): update enhance and similarity phase be torch operation. [83398480](https://github.com/zjykzj/SimpleIR/tree/83398480d66e3a4b76997d03890c06d0ba0a7311)
* Bug fixes
* Breaking changes
  1. refactor(enhancer.py): modify enhance_type "normal" to "identity". [5da1a760](https://github.com/zjykzj/SimpleIR/tree/5da1a760911086e8ca5f675769db3a501fef31d1)
  2. refactor(config): modify _C.METRIC.SIMILARITY_TYPE to _C.METRIC.DISTANCE_TYPE. [4dda521f](https://github.com/zjykzj/SimpleIR/tree/4dda521f3107ec255c7a5f5b9d648b74b72f87ea)
  3. refactor(ghostnet.py): remove "act1" feature. [84150b76](https://github.com/zjykzj/SimpleIR/tree/84150b76e81d36dc57e6d16f1114969a86cc745a)

## v0.4.1

* New features
  1. perf(simpleir): use zcls2's trainer.py. [69e702ba](https://github.com/zjykzj/SimpleIR/tree/69e702ba1699963b56db5608cbaa98a99dc080e0)
  2. fix(trainer.py): fix mixup usgae. [19e259a4](https://github.com/zjykzj/SimpleIR/tree/19e259a44cd257bb8f422f9cdc6450f47b460f82)
* Bug fixes
* Breaking changes.

## v0.4.0

* New features
  1. build(python): update zcls2 ~= 0.4.0. [a3e23c23](https://github.com/zjykzj/SimpleIR/tree/a3e23c23e83fb3d1431d33d6dc2e4a8e6a3f1fc0)
  2. perf(train.py): use Mixup and Resume(). [3abc40a1](https://github.com/zjykzj/SimpleIR/tree/3abc40a1d068944dfc4cf9f478f4eabcc3b85a87)
  3. feat(trainer.py): add Mixup feature. [f4fe5a4d](https://github.com/zjykzj/SimpleIR/tree/f4fe5a4d68dde74f3d18c17c47c012114dab5fb5)
  4. feat(tools): add eval.sh. [f08a8709](https://github.com/zjykzj/SimpleIR/tree/f08a8709f246568568366b3aaaea19a1d8152956)
  5. feat(models): add feat_type for each model. [492310e3](https://github.com/zjykzj/SimpleIR/tree/492310e3598f3bc9c3a411210b1da7064268298b)
* Bug fixes
* Breaking changes.

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
