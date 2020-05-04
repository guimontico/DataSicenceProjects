# Data Science Nanodegree Project 5: Disaster Response Pipeline Webapp

Author: Guilherme Bruno Montico

Start Date: 2020-04-04

## Description

When natural disasters happen, people use twitter to try and get help. However, not all tweets are natural disaster tweets and there isn't a simple way to use key words in order to identify these kinds of texts. So a Machine learning model was created to handle and categorize those messages with the following steps   

1. Create an ETL which cleans the Data   
2. Create a ML pipeline which performs feature extraction and trains a model   
3. Take model and embed it into a webapp    

## Repo Layout

This repo is split into subdirectories:

1. `DataSicenceProjects/DisasterMessageClassifier/DataProcessing_and_ModelTraining` - Contains python notebooks, and files for ML Engineering
2. `webapp` - Contains a webapp, see README inside this directory for instructions

## Packages Used

* `sys`
* `pandas`
* `sqlalchemy`
* `joblib`
* `re`
* `nltk`
* `sklearn`
* `json`
* `plotly`
* `flask`
