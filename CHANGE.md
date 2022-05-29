
# CHANGE

## v0.8.2

* New features
  1. perf(index): update load_feats() use. [5854d8fa5](https://github.com/zjykzj/SimpleIR/tree/5854d8fa5bbf6303539a77a29d88e32c589326d2)
  2. perf(retrieval_images): update retrieval images use. [f1f558ec33](https://github.com/zjykzj/SimpleIR/tree/f1f558ec3350dc73ec61b2f1096d537548e4683b)
  3. perf(eval.py): add one epoch eval time cal. [d3a09ca0e7](https://github.com/zjykzj/SimpleIR/tree/d3a09ca0e7389ec793367a97480af6243389f3dd)
  4. perf(eval.py): upgrade pretrained model use. [2ce4c00a9](https://github.com/zjykzj/SimpleIR/tree/2ce4c00a9fa53d7ee6e78a73a34ff5672604a865)
  5. perf(index): update max_num and gallery_dict use. [a437e9ef4](https://github.com/zjykzj/SimpleIR/tree/a437e9ef49a4efa67eb3a881eeb271bd5ec1c623)
* Bug fixes
  1. fix(extract_features.py): fix Extractor import path. [121b466c4](https://github.com/zjykzj/SimpleIR/tree/121b466c4d0724d32811dfe1d1f0f2b24f3775a1)
* Breaking changes

## v0.8.1

* New features
  1. perf(data): explicitly set val_sampler=None. [c46bc963](https://github.com/zjykzj/SimpleIR/tree/c46bc9635d5bb8c975b324a28ea1cd9f5f06c6c0)
  2. perf(index): init/clean gallery_dict each epoch and update single mode. [e1d8c73a](https://github.com/zjykzj/SimpleIR/tree/e1d8c73a56204b77373951509e52ef45b253c21d)
  3. feat(eval.py): new. [2ee957e2f](https://github.com/zjykzj/SimpleIR/tree/2ee957e2f26e39909505cc22de90369c44bb7741)
* Bug fixes
* Breaking changes
  
## v0.8.0

* New features
  1. build(python): upgrade zcls2 ~= 0.4.3 to zcls2 ~= 0.4.4. [c70132c1ac](https://github.com/zjykzj/SimpleIR/tree/c70132c1ac8ad163ba2c63a95114d5a1dd8156ed)
  2. feat(index): create IndexMode for different way to index. [7ba156c2](https://github.com/zjykzj/SimpleIR/tree/7ba156c267f5d4a90ee75abda71de49349f45d8a)
  3. perf(extract_features.py): add feats extract choice for gallery set or query set. [24ea2184](https://github.com/zjykzj/SimpleIR/tree/24ea2184bec070b11808707baae31e78346db8d9)
  4. perf(extract): update ExtractHelper use. [95c4ef0ea](https://github.com/zjykzj/SimpleIR/tree/95c4ef0ea59756e9161afcdd4b7daff22e1947f6)
* Bug fixes
* Breaking changes
  1. refactor(index): update. [fceb900e52796a](https://github.com/zjykzj/SimpleIR/tree/fceb900e52796aff3036a1db165124ed82268a17)
  2. refactor(extract): modify ExtractHelper to Extractor. [b1806547](https://github.com/zjykzj/SimpleIR/tree/b1806547cc8fe1e78879790a2a0d584864b9792c)
  3. refactor(configs): modify _C.EVAL.INDEX.TRAIN_DIR to GALLERY_DIR and add _C.EVAL.FEATURE.QUERY_DIR. [ee628cd](https://github.com/zjykzj/SimpleIR/tree/ee628cdcfd5f1c7015d233181106c26b6367ad86)
  4. refactor(simpleir): refactor configs and metric module. [ac8ba8fe06](https://github.com/zjykzj/SimpleIR/tree/ac8ba8fe061b1c8b2fbd6e91b0223de5d271240e)

## v0.7.0

* New features
  1. feat(configs): add _C.METRIC.EVAL.EVAL_TYPE. [55980d186c](https://github.com/zjykzj/SimpleIR/tree/55980d186cc22ef05797cef026a8449fe32f43a3)
  2. feat(index): add train_dir use. [4561f9220](https://github.com/zjykzj/SimpleIR/tree/4561f9220553ade8b6e41d2c80b80249797272b0)
  3. perf(simpleir): new ACCURACY impl. [3bc52d7b3](https://github.com/zjykzj/SimpleIR/tree/3bc52d7b313a2390a5155daf711cff5465807128)
  4. feat(metric): new EvaluateHelper. [e37e02db4](https://github.com/zjykzj/SimpleIR/tree/e37e02db4069b67f955f83da179c19c5232435a5)
* Bug fixes
* Breaking changes
  1. feat(configs): add _C.METRIC.INDEX.TRAIN_DIR and refactor _C.METRIC use. [a3bb4924d8](https://github.com/zjykzj/SimpleIR/tree/a3bb4924d83f5306aafe7a08c54223266af5f47c)

## v0.6.3

* New features
  1. perf(infer.py): use cfg.METRIC.ENHANCE_TYPE. [a19a90641fac](https://github.com/zjykzj/SimpleIR/tree/a19a90641faceba406cce95f40b0ea7f01d7780b)
  2. perf(make_query_gallery_set.py): support GeneralSplitter. [b03350008](https://github.com/zjykzj/SimpleIR/tree/b03350008d822d691acfb6beb7ca511fb5a27366)
  3. feat(split): add GeneralSplitter. [ae5c0b0cc](https://github.com/zjykzj/SimpleIR/tree/ae5c0b0cc9c044280be4197e98e01bb7f154ed6b)
  4. perf(extract): update Extract Features module use. [cdff10f](https://github.com/zjykzj/SimpleIR/tree/cdff10fd4e7b4af03f37f9cc32e68c411b0f7682)
  5. perf(dataset): add w_path support. [42f696be](https://github.com/zjykzj/SimpleIR/tree/42f696be1e3e6c10b38d0e2d61397b592668dff6)
  6. feat(dataset): add GeneralDataset. [07fd67a1](https://github.com/zjykzj/SimpleIR/tree/07fd67a19e4aa425f22dfca697d344ec34c7534d)
* Bug fixes
* Breaking changes

## v0.6.2

* New features
  1. build(python): upgrade zcls2 ~= 0.4.2 to zcls2 ~= 0.4.3. [068f4b2b](https://github.com/zjykzj/SimpleIR/tree/068f4b2b0cab72646c788081a691cc50f135b80e)
  2. feat(data): for INFER stage, always get (image, target, img_path). [cf994b4de5](https://github.com/zjykzj/SimpleIR/tree/cf994b4de56d688fa09219b4f39d4305e3cb3322)
* Bug fixes
  1. fix(configs): fix NORMALIZE use. [61422de5c](https://github.com/zjykzj/SimpleIR/tree/61422de5c584f89463c4d6749a99a30a43548b23)
  2. fix(index): error skipping rank/re_rank stage. [c21bfca](https://github.com/zjykzj/SimpleIR/tree/c21bfca3cac21f96b4c06be53923ced9ff45c7ca)
* Breaking changes

## v0.6.1

* New features
  1. build(python): upgrade zcls2 ~= 0.4.1 to zcls2 ~= 0.4.2. [ec055010d](https://github.com/zjykzj/SimpleIR/tree/ec055010d88f7dac8ac68ceefae79919ddf45f87)
  2. refactor(configs): update NORMALIZE usage. [0e65e2540a0](https://github.com/zjykzj/SimpleIR/tree/0e65e2540a00da1e8167644f394e2eefbf03a209)
* Bug fixes
* Breaking changes

## v0.6.0

* New features
  1. feat(metric): add RankType and ReRankType for ranker. [dcbeae6b2](https://github.com/zjykzj/SimpleIR/tree/dcbeae6b2a92fc9f15223a82164bae837d4fcd8f)
  2. feat(metric): add Enum DistanceType for distancer. [506092e03](https://github.com/zjykzj/SimpleIR/tree/506092e034ea9172686f048b42f94c1700f37012)
  3. feat(metric): add Enum EnhanceType for enhancer. [00172cd15d9](https://github.com/zjykzj/SimpleIR/tree/00172cd15d9ce2dc029b111414c20117485ee8c4)
  4. feat(metric): add Enum AggregateType for aggregator. [2f2696f4e](https://github.com/zjykzj/SimpleIR/tree/2f2696f4e1b4d5187948aaefb1a7a2986a438102)
  5. feat(metric): update Feature module use. [a205c84](https://github.com/zjykzj/SimpleIR/tree/a205c8473e90ce8f7ded8eabae2efd9535c0a2ba)
  6. feat(metric): create Feature module, including Aggregate and Enhance module. [0b9dafe55](https://github.com/zjykzj/SimpleIR/tree/0b9dafe555dfd3ecc5d5f23111a695f2395b85ab)
  7. feat(metric): create Index module, including rank and re_rank. [8503922f9](https://github.com/zjykzj/SimpleIR/tree/8503922f92b6326a9f0523794a4025750d350570)
  8. feat(metric): add ReRank module. [7c50ee](https://github.com/zjykzj/SimpleIR/tree/7c50ee31ddd41b6167cf3333b737b91bbeddc164)
  9. feat(mkdocs): init. [bc169e7dbb](https://github.com/zjykzj/SimpleIR/tree/bc169e7dbbfd5aabcaa92dc505f2fbcb735e78c0)
* Bug fixes
* Breaking changes

## v0.5.3

* New features
  1. feat(models): add mobilenet_v3_large / mobilenet_v3_small. [5afd5a457](https://github.com/zjykzj/SimpleIR/tree/5afd5a457c641bd999cc8ac0dfd390c2c8d051ed)
* Bug fixes
* Breaking changes

## v0.5.2

* New features
  1. perf(ranker): update knn_rank use. [3b77d172](https://github.com/zjykzj/SimpleIR/tree/3b77d172048c357173ba5de0acb738f7776a4068)
  2. feat(ranker): add KNN Rank. [d167eedd9](https://github.com/zjykzj/SimpleIR/tree/d167eedd93664db9ae783dd8013cdcab735bc8e4)
* Bug fixes
* Breaking changes

## v0.5.1

* New features
* Bug fixes
  1. fix(resnet.py): fix ResNet forward() use. [a1b55b6](https://github.com/zjykzj/SimpleIR/tree/a1b55b6ac183600aaeb0f1aea215592e849cc047)
* Breaking changes

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
