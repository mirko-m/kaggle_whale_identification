# kaggle_whale_identification
This repository contains my code for the Kaggle humpback whale identification challenge. Given images of whale flukes the challenge is to predict the identity of the whales. My current approach to this challenge is to use transfer learning and a pretrained neural network (NN) of the Resnet18 architecture.

My analysis is contained in the following files : 
- **data_exploration.ipynb**: My data exploration shows that the distribution of the different categories (aka classes) is strongly skewed. The training set contains close to 10000 images and about 4000 different labels. Many of the labels appear only once, while the most common label 'new_whale' appears 810 times and the second most common label appears 34 times. 
- **toy_model.ipynb**: To address the problem of the skewed classes I first train a toy-model on a subset consisting of 56 whales which is not skewed. The purpose of this model is to learn what sort of features are relevant for the identification of whales.
- **full_model.ipynb**: The full model uses the hidden layers of the toy-model, only the final layer is changed. The hope is that the toy-model has learned useful features which can be used to classify all of the whales.
- **ful_model_gpu.ipynb**: Similar to the notebook full_model.ipynb, but I used a p2.xlarge GPU instance from AWS to train the full model even further. This gives me a better training loss and accuracy. The speedup due to the GPU is roughly a factor 60.
- **predict.ipynb** Makes predictions on the test set and stores them in the file my_submission.csv which can be uploaded to Kaggle.

Furthermore some usful functions and classes for interacting with pytorch can be found in **my_utils.py**. Some of the code here was adapted from Refs [1] and [2], while many of the ideas are taken from Ref [3].

## Ideas for improvements
- Come up with a good way to create a validation set when training on the full data. At the moment I am not using a validation set here, because many of the whales only appear once in the training set and I don't want to remove them from the training set.
- Come up with a better way to treat the "new_whale" class.
- Use oversampling for those whales which only appear a few times in the training set.
- Implement a triplet loss function for the full model.

## References
[1] Pytorch data loading tutorial https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

[2] Pytorch transfer learning tutorial https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

[3] fast.ai MOOC http://www.fast.ai/
