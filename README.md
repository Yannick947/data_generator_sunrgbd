# data_generator_sunrgbd
Data generator to bring the labels of the SUNRGBD labels ([SUNRGBD dataset](http://3dvision.princeton.edu/projects/2015/SUNrgbd/)) into the format for training [Mask-RCNN of matterport](https://github.com/matterport/Mask_RCNN) to perform instance segmentation.

The repo provides two main functionalities: 
 - Class dimension reduction
 - Label file generation 

## Class dimension reduction

When parsing all label files provided within the SUN dataset more than 14k classes occur which can be reduced to reduced set with this repo. The utilities for removing typos and performing a word2vec approach to reduce the number of classes are provided in the folder "class_dimension_reduction".

A exemplary class matching for the class bookcase can be seen in this Figure: ![Class matching for bookcase](https://github.com/Yannick947/data_generator_sunrgbd/images/classes_matching.png)

Some manual cleaning can be done with funcitonalities provided in the training repo for [Mask-RCNN](https://github.com/Yannick947/Mask_RCNN).

## Label file generation for Mask-RCNN

The parsing of all label files and using the prior extraction of classes is done by calling the sunrgbd_generator/generator.py function. The transformation class in sunrgbd_generator/sunrgbd_to_maskrcnn.py provides custom transformations for the specific format needed in Mask-RCNN. This class can be changed to fit other input format needs. 


## Train on rgbd data with Mask-RCNN

The repo for training utilities of Mask-RCNN with the provided data format can be found [here](
https://github.com/Yannick947/Mask_RCNN). The network in this fork from matterport was already altered to provide the option to train on RGBD data. 