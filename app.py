import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


df = pd.read_csv('https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/doctolib_simplified_dataset_01.csv')

# Split dataset into X features and Target variable
df.drop(columns=['Unnamed: 0', 'PatientId', 'AppointmentID'], axis=1, inplace=True)
df[['ScheduledDay', 'AppointmentDay']] = df[['ScheduledDay', 'AppointmentDay']].apply(pd.to_datetime)
df['No-show'] = df['No-show'].map({'Yes': 1, 'No': 0})
df['Gender'] = df['Gender'].map({'F': 1, 'M': 0})
df['ApppointmentWeekday'] = df['AppointmentDay'].dt.dayofweek
df['ScheduledDayWeekday'] = df['ScheduledDay'].dt.dayofweek
df = pd.get_dummies(data=df, columns=['Neighbourhood'], drop_first=True)

cols = df.columns.tolist()
del cols[10]
del cols[1:3]

X = df[cols]
y = df['No-show']

# Split our training set and our test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Set your variables for your environment
EXPERIMENT_NAME="mlflow-cancellation-prediction-xgboost"

# Instanciate your experiment
client = mlflow.tracking.MlflowClient()

# Set experiment's info 
mlflow.set_experiment(EXPERIMENT_NAME)

# Get our experiment info
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
run = client.create_run(experiment.experiment_id) # Creates a new run for a given experiment

# Call mlflow autolog
mlflow.sklearn.autolog()

with mlflow.start_run(run_id = run.info.run_id):

    # Instanciate and fit the model 
    xgb = XGBClassifier(max_depth=8, min_child_weight=10, n_estimators=100)
    xgb.fit(X_train.values, y_train.values)

    # Store metrics 
    predicted_qualities = xgb.predict(X_test.values)
    accuracy = xgb.score(X_test.values, y_test.values)

    # Print results 
    print("XGBoost model")
    print(f"Accuracy: {accuracy}")
    
    # Log Metric 
    mlflow.log_metric("Accuracy", accuracy)