# Disaster Response Pipeline Project
### Project Description:

In this project, I built a machine learning pipeline built into a flask application. The project include a web app where an emergency worker can input a message and get classification results.

### File Descriptions:
The project contains the following files,

* ETL Pipeline Preparation.ipynb: Notebook for the ETL pipelines
* ML Pipeline Preparation.ipynb: Notebook for the machine learning pipelines
* data/process_data.py: The ETL pipeline used to process data in preparation for model building.
* models/train_classifier.py: The Machine Learning pipeline used to fit, improve, evaluate, and export the model to a Python pickle.
* app/templates/~.html: HTML pages for the web app.
* run.py: Start the Python server for the web app and prepare visualizations.
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
