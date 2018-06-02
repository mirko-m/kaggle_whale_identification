# kaggle_whale_identification
This repository contains my code for the Kaggle humpback whale identification challenge. Given images of whale flukes the challenge is to predict the identity of the whales. My current approach to this challenge is to use transfer learning and a pretrained neural network (NN) of the Resnet18 architecture.

My analysis is contained in the following files : 
- **data_exploration.ipynb**: My data exploration shows that the distribution of the different categories (aka classes) is strongly skewed. The training set contains close to 10000 images and about 4000 different labels. Many of the labels appear only once, while the most common label 'new_whale' appears 810 times and the second most common label appears 34 times. 
- **toy_model.ipynb**: To address the problem of the skewed classes I first train a toy-model on a subset consisting of 56 whales which is not skewed. The purpose of this model is to learn what sort of features are relevant for the identification of whales.
- **full_model.ipynb**: The full model uses the hidden layers of the toy-model, only the final layer is changed. The hope is that the toy-model has learned useful features which can be used to classify all of the whales.

Furthermore some usful functions and classes for interacting with pytorch can be found in **my_utils.py**. Some of the code here was adapted from Refs [1] and [2], while many of the ideas are taken from Ref [3].

## References
[1] Pytorch data loading tutorial https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

[2] Pytorch transfer learning tutorial https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

[3] fast.ai MOOC http://www.fast.ai/
