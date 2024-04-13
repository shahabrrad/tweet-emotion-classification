# Emotion Detection from Tweets - CS577 Homework 1

## Overview
This repository contains the implementation of a logistic regression 
classifier and a multi-layer neural network to detect emotions from 
Tweets. This homework is part of CS577 at Purdue University, focusing on 
building these models from scratch without using built-in functions from 
libraries like sklearn for the machine learning part.

The script will train the logistic regression and neural network models on the training data and predict emotions on the test data. The predictions will be saved in `test_lr.csv` and `test_nn.csv` for logistic regression and neural network models, respectively.

## Dataset
The dataset consists of 1200 labeled Tweets for training and 800 Tweets 
for testing. The Tweets are annotated with one of six emotions: joy, love, 
sadness, anger, fear, surprise. The test dataset does not include labels.



## Implementation Details
### Logistic Regression
- The `LR()` function in `main.py` learns a logistic regression classifier using cross-validation and outputs predictions in `test_lr.csv`.

### Multi-layer Neural Network
- The `NN()` function in `main.py` learns a multi-layer neural network classifier using cross-validation and outputs predictions in `test_nn.csv`.

Both functions ensure that the ID-to-text-to-emotion mapping is preserved as required.

## Advanced Text Preprocessing
- The code includes options for bag of words features and word embeddings. For word embeddings, pre-trained vectors should be averaged to create a feature representation for each tweet.

## Submission Instructions
Submit your code via Turnin as detailed in the assignment description. Ensure that only `main.py`, `test_lr.csv`, and `test_nn.csv` are in the submission folder named `yourusername_hw1`.

## Contact
For any queries regarding the code or the homework, please [create an issue](https://github.com/yourusername/cs577-hw1/issues) in this repository.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
