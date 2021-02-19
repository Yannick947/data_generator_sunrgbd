# SUNRGBD data generator
Data generator to bring the labels of the [SUNRGBD dataset](http://3dvision.princeton.edu/projects/2015/SUNrgbd/) into the format for training [Mask-RCNN of matterport](https://github.com/matterport/Mask_RCNN) to perform instance segmentation.

The repo provides two main functionalities: 
 - Class dimension reduction
 - Label file generation 

## Class dimensionality reduction

When parsing all label files provided within the SUNRGBD dataset more than 14k classes occur which can be reduced to a smaller set with this repository. The utilities for removing typos and performing a word2vec approach to reduce the number of classes are provided in the folder "class_dimension_reduction". The word2vec approach uses a pretrained linguistic model to transform the labels into a vector. By defining some fix classes which one wants to be able to train on, it is possible to match all similar classes into these predefined classes. This can be particulary useful in this dataset since the labels are created by different labelers and with different "original" label classes. Thus classes like "table" and "desk" might be considered as different even there is no initial purpose of putting them into different classes. This makes it impossible for an algorithm to distinguish between them and this problem is tackled by this repository by merging these identical label classes. 

An exemplary class matching for the class bookcase can be seen in this Figure: ![Class matching for bookcase](https://github.com/Yannick947/data_generator_sunrgbd/blob/main/images/classes_matching.png)

Since there might be some bad matches as well, manual cleaning can be done with funcitonalities provided in the training repo for [Mask-RCNN](https://github.com/Yannick947/Mask_RCNN).

A threshold can be set in class_dimension_reduction/spacy_dimension_reduction.py to increase the reliablility that classes are "similar enough". The library which is used to perform the word2vec approach is the [spacy](https://spacy.io/) library.

## Label file generation for Mask-RCNN

The parsing of all label files and using the prior extraction of classes is done by calling the sunrgbd_generator/generator.py function. The transformation class in sunrgbd_generator/sunrgbd_to_maskrcnn.py provides custom transformations for the specific format needed in Mask-RCNN. This class can be changed to fit other input format needs. 


## Train on rgbd data with Mask-RCNN

The repo for training utilities of Mask-RCNN with the provided data format can be found [here](
https://github.com/Yannick947/Mask_RCNN). The network in this fork from matterport was already altered to provide the option to train on RGBD data. Utilities for evaluation and data augmentation for images with and without the depth channel are provided. 