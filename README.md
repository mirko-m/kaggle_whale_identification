# kaggle_whale_identification
This repository contains my code for the Kaggle humpback whale identification challenge. Given images of whale flukes the challenge is to predict the identity of the whales. My current approach to this challenge is to use transfer learning and a pretrained neural network (NN) of the Resnet18 architecture.

My analysis is contained in the following files : 
- **data_exploration.ipynb**: My data exploration shows that the distribution of the different categories (aka classes) is strongly skewed. The training set contains close to 10000 images and about 4000 different labels. Many of the labels appear only once, while the most common label 'new_whale' appears 810 times and the second most common label appears 34 times. 
- **toy_model.ipynb**: To address the problem of the skewed classes I first train a toy-model on a subset consisting of 56 whales which is not skewed. The purpose of this model is to learn what sort of features are relevant for the identification of whales.
- **full_model.ipynb**: The full model uses the hidden layers of the toy-model, only the final layer is changed. The hope is that the toy-model has learned useful features which can be used to classify all of the whales.
- **ful_model_gpu.ipynb**: Similar to the notebook full_model.ipynb, but I used a p2.xlarge GPU instance from AWS to train the full model even further. This gives me a better training loss and accuracy. The speedup due to the GPU is roughly a factor 60.
- **predict.ipynb** Makes predictions on the test set and stores them in the file my_submission.csv which can be uploaded to Kaggle.
- **duplicate_images.ipynb** There is a thread on the Kaggle forum discussing that some of the images in the training set appear multiple times. This notebook identifies these images using a perceptual hash algorithm from ImageHash [4,5]. This notebook also creates a csv file with a subset of unique images for the training set. 10 of the images appear with more than one label and are therefore thrown out.
- **predict_phash.ipynb** Some of the images from the training set also appear in the test set. This notebook makes a prediction by finding these images and predicting new_whale for all other images.
- **predict_new_whale_as_extra_class.ipynb** Remove all of the new_whales from the training set and then retrain the final layer of the full model. Once this is done the new_whale class is taken into consideration by defining a threshold and predicting new_whale when the probability for all other labels is smaller than said threshold. A threshold of 0.2 seems to work well (based on the score I get when submitting to Kaggle, be careful of fitting to the test set). A validation set containing all of the new_whale pictures and some of the more common whales is used. Predictions from this model are combined with those from predict_phash.ipynb.
- **predict_v2.ipynb** Used for testing.

Furthermore some usful functions and classes for interacting with pytorch can be found in **my_utils.py**. Some of the code here was adapted from Refs [1] and [2], while many of the ideas are taken from Ref [3].

## Ideas for improvements
- Find a better way to set the threshold for the new_whale class.
- Use oversampling for those whales which only appear a few times in the training set.
- Preprocess Images:
  * Remove text from images.
  * Crop images to fluke only.
  * Use more data augmentation.
  * Try converting all images to grayscale.
- Implement a triplet loss function for the full model and toy model.

## References
[1] Pytorch data loading tutorial https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

[2] Pytorch transfer learning tutorial https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

[3] fast.ai MOOC http://www.fast.ai/

[4] ImageHash https://pypi.org/project/ImageHash/

[5] Perceptual hash algorithm http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
