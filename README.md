# MSc_Project

##Overview
This project implements a deep-learning-based method for predicting the axial focal position of larval zebrafish from microscopy images. The aim is to provide rapid focus estimation that could eventually be used to control an electrically tuneable lens (ETL) during imaging of freely moving zebrafish.
The main model used is MobileNetV2, trained on zebrafish Z-stack image patches to predict:
Focal distance magnitude — regression output
Focal direction — classification output (negative/positive)
The two outputs are combined to obtain a signed focal-distance prediction.
#Model
The MobileNetV2 architecture is modified to accept single-channel grayscale microscopy images and produce three outputs:
Focal-distance magnitude
Negative focal direction
Positive focal direction
The predicted magnitude is scaled by the Z-stack spacing used during training.
#Dataset
The dataset consists of image patches extracted from zebrafish Z-stacks with known focal distances. Each image is associated with:
Image filename
Focal distance (µm)
Focal direction
Separate datasets are used for training and testing, with unseen fish reserved for evaluation.
Training
The model is trained using:
Optimizer: Adam
Regression loss: Mean Squared Error (MSE)
Classification loss: Cross Entropy
Total loss: Regression loss + classification loss
Data augmentation can include random affine transformations such as rotation, translation, scaling and shearing.
##Evaluation
Model performance is evaluated using:
Regression: Mean Absolute Error (MAE)
Classification: Classification accuracy
Inference time is also measured to assess the potential for real-time focus control using an ETL.
