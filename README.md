# Spaceship-Titanic on Kaggle

## Introduction

This is my attempt to solve the Spaceship Titanic problem on Kaggle. The problem is to predict whether a passenger was transported to another dimension or not. The data is provided by Kaggle and can be found [here](https://www.kaggle.com/c/Spaceship-Titanic/data). 

All the models that I have used are from the [scikit-learn](https://scikit-learn.org/stable/) library. For consistency, the same data preprocessing steps were used for all the models. 


## Files

`DownloadData.py` - Downloads the data from Kaggle

`AdaBoost.py` - AdaBoost classifier

`Bagging.py` - Bagging classifier

`ExtraTrees.py` - ExtraTrees classifier

`GBM.py` - Gradient Boosting classifier

`KNN.py` - K-Nearest Neighbors classifier

`LogReg.py` - Logistic Regression classifier 

`Naive_Bayes.py` - Naive Bayes classifier

`Neural.py` - Neural Network classifier

`Random_Forest.py` - Random Forest classifier

`Stacking.py` - Stacking classifier

`SVM.py` - Support Vector Machine classifier

`Tree.py` - Decision Tree classifier

`Voting.py` - Voting classifier

`Visualization.ipynb` - Jupyter notebook for visualization of the data

## Running the code

First, download the data from Kaggle either manually or by running `DownloadData.py`. Then, run any of the classifiers to get the predictions. 

## Results

The best result was achieved by the AdaBoost followed by the Gradient Boosting classifier. 

The top 5 models are as follows:

| Model | Accuracy |
| --- | --- |
| AdaBoost | 0.79541 |
| Gradient Boosting | 0.79237 |
| Bagging | 0.79191 |
| SVM | 0.79074 |
| Voting | 0.79074 |



