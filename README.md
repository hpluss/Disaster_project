# Disaster Response Pipeline Project

### Summary
The purpose of this project is to build a model that categorizes the messages received during real-life disasters events. This will help to better target the assistance operations to be deployed.
The final product of the project is a web application on which messages are entered and the model generates the relevant categories.
The model was developed using NLP (natural language processing) algorithms.


### Project structure

    app
    | - template
    | |- master.html # main page of web app
    | |- go.html # classification result page of web app
    |- run.py # Flask file that runs app
    data
    |- disaster_categories.csv # data to process
    |- disaster_messages.csv # data to process
    |- process_data.py  # the cleaning data code
    |- DisasterResponse.db # database to save clean data to
    models
    |- train_classifier.py  # the model building code
    |- classifier.pkl # saved model
    README.md
    
### Dependencies

-	Python 3.5+
-	Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
-	Natural Language Process Libraries: NLTK
-	SQLlite Database Libraqries: SQLalchemy
-	Model Loading and Saving Library: Pickle
-	Web App and Data Visualization: Flask, Plotly


### Instruction
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
