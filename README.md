# Thesis

This repository contains the code and resources for our research. Below is an overview of the folder structure and the key components.

## Model

The main model, referred to as PTM in our article, is located inside the `Model/Model` directory. This directory contains the implementation of the model along with the necessary scripts to train and evaluate it.

## Configuration

The configuration settings for the model are defined in the `config` class. These settings include parameters such as learning rate, batch size, number of epochs, hidden layers, and more.

## Running the Model

We provide two options to run the model:

1. **Run with K-Fold Cross-Validation**: This option allows you to train and evaluate the model using K-Fold cross-validation. The relevant code is located in the `Model/run_k_fold.py`.

2. **Run with Train, Validation, and Test Split**: This option allows you to train the model on a training set, validate it on a validation set, and test it on a test set. The relevant code is located in the `Model/run_train_val_test.py`.

## Data

The `Data` folder contains scripts and resources to generate the dataset type required for training and evaluating the model. Ensure that the dataset is properly formatted and placed in the appropriate directory before running the model.

## Programming Concepts
This JSON file contains the final programming concepts and details on how concepts are merged in each dataset.

## Model Architecture

An image of our model architecture will be placed here.

![Model Architecture](path/to/your/image.png)