# Diabetes Prediction 
![project8](https://github.com/user-attachments/assets/9f4c45e4-09a8-442e-8cab-cb189c94e65d)

## Overview
This project is a Diabetes Prediction Application designed to predict the likelihood of diabetes based on clinical features. The project includes machine learning model development, hyperparameter tuning, and deployment as a web application. The application is containerized using Docker and hosted on AWS EC2. Performance metrics and model comparisons were tracked using MLflow and DagsHub.

## Features
- ### Exploratory Data Analysis (EDA): 
Identified patterns and engineered new features for better model performance.
- ### Hyperparameter Tuning:
Utilized Grid SearchCV to optimize the model parameters for multiple algorithms.
- ### Model Performance Tracking:
Visualized model performance using MLflow and DagsHub.
- ### Web Application:
Built a Flask-based app to take user input, perform feature engineering, and provide predictions.
- ### Dockerized Deployment: 
The application is containerized for consistent and portable deployment.
- ### Cloud Hosting: 
Hosted the application on an AWS EC2 instance for public accessibility


## Requirements
- ### Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

- ### Setup and Deployment
- __1. Clone the Repository__
```bash
git clone https://github.com/your-repo/diabetes-prediction.git
cd diabetes-prediction
```

- __2. Install Dependencies__
Ensure you have Python installed and run:

```bash
pip install -r requirements.txt
```

- __3. Run the Application Locally__
```bash
python app.py
```

Access the application at `http://127.0.0.1:8000`.

- __4. Build and Run Docker Container__
```bash
docker build -t diabetes_prediction .
docker run -p 8000:8000 diabetes_prediction
```

- __5. Deploy on AWS EC2__
- Set up an AWS EC2 instance.
- Install Docker and pull the Docker image to the instance.
- Run the application using the above Docker commands.

Running @ http://65.2.121.33:8000/ {Or May be not: Exhausted free monthly quota :/ }

## Technologies Used
- __Flask:__ Backend web framework.
- __Docker:__ Application containerization.
- __AWS EC2:__ Cloud hosting.
- __MLflow:__ Experiment tracking and model registry.
- __DagsHub:__ Collaboration and tracking platform.
- __Pandas & Scikit-Learn:__ Data processing and machine learning.
