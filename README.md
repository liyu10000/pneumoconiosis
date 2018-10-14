## CheXNet Exploration

### Source of the ChestX-ray14 paper and data
 - __original paper__: [https://arxiv.org/pdf/1705.02315.pdf](https://arxiv.org/pdf/1705.02315.pdf)
 - __paper by NG__: [https://arxiv.org/abs/1711.05225](https://arxiv.org/abs/1711.05225)
 - __dataset_nih__: [https://nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC)
 - __dataset_kaggle__: [https://www.kaggle.com/nih-chest-xrays/data](https://www.kaggle.com/nih-chest-xrays/data)


### Analysis of dataset
 - __concerns from a radiologist__: [Exploring the ChestXray14 dataset: problems](https://lukeoakdenrayner.wordpress.com/2017/12/18/the-chestxray14-dataset-problems/)

### Model implementation
Since the models described in source paper and paper by NG's group are not available, I did some search and located some useful dataset analyze and model exploration efforts made by people on Kaggle and Github.
 - _Kevin Mader_ on Kaggle. He has presented very good dataset analysis, data preprocessings and [model training](https://www.kaggle.com/kmader/cardiomegaly-pretrained-vgg16/notebook). 
 - _Caleb P_ on Kaggle. Tried to MobileNet and InceptionResNetV2 as base model and showed sample [data training and result presenting](https://www.kaggle.com/cpagel/adjust-simple-xray-cnn/notebook).
 - _arnoweng_ on GitHub. It is a [pytorch reimplementation](https://github.com/arnoweng/CheXNet) of CheXNet, that presented by NG paper. It only has prediction code, not training code.
 - _brucechou1983_ on GitHub. It is a [keras reimplementation](https://github.com/brucechou1983/CheXNet-Keras) of CheXNet. Contains full codes. 


### My exploration
 - _cnn1, cnn2_: used the data preprocessing and model building method described by Kevin Mader.
 - _cnn3_: used data preprocessing method by Kevin Mader and model building method by Caleb P.
 - _cnn4_: used data preprocessing method by Kevin Mader and model building method by brucechou1983.
 - _cnn5_: used data preprocessing method by Kevin Mader and model building method by brucechou1983. Updated image proprocessing method: center cropping, 0-1 normalization, mean/std normalization. Train with 14 classes.


#### Useful links
 - keras checkpoint saving: https://machinelearningmastery.com/check-point-deep-learning-models-keras/