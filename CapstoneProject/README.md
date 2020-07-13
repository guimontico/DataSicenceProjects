# Data Science Nanodegree Project: Logistic Regression with Spark

Author: Guilherme Bruno Montico

Data: 2020-07-11

## Description

In this Project I will predict user churn of "sparkify", a music streaming serivce.

Blog post for this project can be found [here](https://medium.com/@guilhermebmontico/using-spark-to-analysis-music-streaming-data-e4ecb2d00f75?sk=631ccd9568a6683ba53c3a9ab96f61ea).

The objective of this project is to apply Spark knowledge and create a product that can be deployed with large cloud datasets applying the techniques developed in the course and in extra curricular sessions

The data subset shows the usage of the application's subscribers for approximately three months. The goal was to discover the high-risk group of customers who are likely to churn.

### Data Descriptions

The data provided is the user log of the service, having demographic info, user activities, timestamps and etc. The data contains the user information logs that includes 

* Add Friend
* Add to Playlist
* Cancel/Cancel Confirmation
* Submit Upgrade/Upgrade
* Submit Downgrade/Downgrade
* Error
* Help
* Home
* Logout
* Nextsong
* Roll Advert
* Save Settings
* Thumbs Up / Down

## Resulty Summary

The result generated was a list of users that can be used in analyzes to find ways to maintain their subscription. The f1 model score was close to 68% for logistic regression, with an accuracy of 74%. It is expected that with the complete data the results can be better

## Repo Layout

The repo layout in this instance is relatively simple. The work done can be found in the jupyter notebook, Sparkify_project.ipynb. The json file which contains the data was a little too big to commit and so I have ommitted it.


## Packages Used

* `pyspark`
* `time`
* `matplotlib`
* `numpy`
* `pandas`
* `seaborn`


## Acknowledgements

N/A
