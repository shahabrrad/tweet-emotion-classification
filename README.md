# Emotion Detection from Tweets

## Overview
This repository contains the implementation of a logistic regression 
classifier and a multi-layer neural network to detect emotions from 
Tweets. This homework is part of the NLP course, focusing on 
building these models **from scratch** without using built-in functions from 
libraries like sklearn for the machine learning part.

The script will train the logistic regression and neural network models on the training data and predict emotions on the test data. The predictions will be saved in `test_lr.csv` and `test_nn.csv` for logistic regression and neural network models, respectively.

## Dataset
The dataset consists of 1200 labeled Tweets for training and 800 Tweets 
for testing. The Tweets are annotated with one of six emotions: joy, love, 
sadness, anger, fear, surprise. The test dataset does not include labels.



## Implementation Details
### Logistic Regression
- The `LR()` function in `main.py` learns a logistic regression classifier using cross-validation and outputs predictions in `test_lr.csv`.
- Loss Function: Cross-Entropy with L1 regularization.
- Feature Representation: TF-IDF values for each word.

### Multi-layer Neural Network
- The `NN()` function in `main.py` learns a multi-layer neural network classifier using cross-validation and outputs predictions in `test_nn.csv`.
- Architecture: One hidden layer with 50 nodes. Input layer based on TF-IDF value dimensions and output layer for one-hot encoded emotions.
- Activation Functions: ReLU for the hidden layer and softmax for the output layer.
- Loss Function: Cross-Entropy combined with L1 or L2 regularization.

Both functions ensure that the ID-to-text-to-emotion mapping is preserved as required.

## Model Details and Hyperparameters
### Hyperparameter Tuning
- **Logistic Regression**: Learning rate and regularization were tuned using 5-fold cross-validation and grid search.
- **Neural Network**: Learning rate, regularization rate and type, number of hidden nodes, and batch size were tuned. Observations on overfitting were noted based on training and validation loss comparisons.
- Detailed learning curves were used to decide the best parameters and to observe the effects of different hyperparameter settings on model performance and overfitting.


## Contact
For any queries regarding the code or the homework, please [create an issue](https://github.com/yourusername/cs577-hw1/issues) in this repository.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
