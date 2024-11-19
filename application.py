import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

# Load the model
modelpath = "mlruns/917724354246876635/d1c8de9ce7194163a710accc004dd368/artifacts/model/model.pkl"
with open(modelpath, "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

# Define the expected feature names based on the training data
expected_features = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age', 'NewBMI_Obesity 1',
    'NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight',
    'NewBMI_Underweight', 'newInsulin_Normal', 'NewGlucose_Low',
    'NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret'
]

# Feature engineering for Glucose, Insulin, and BMI
def transform_features(data):
    # Convert necessary fields to appropriate numeric types
    data['Pregnancies'] = data['Pregnancies'].astype(int)
    data['BloodPressure'] = data['BloodPressure'].astype(int)
    data['SkinThickness'] = data['SkinThickness'].astype(int)
    data['DiabetesPedigreeFunction'] = data['DiabetesPedigreeFunction'].astype(float)
    data['Age'] = data['Age'].astype(int)
    data['Glucose'] = data['Glucose'].astype(float)
    data['Insulin'] = data['Insulin'].astype(float)
    data['BMI'] = data['BMI'].astype(float)
    
    data['NewGlucose'] = 'Secret'
    data.loc[data['Glucose'] <= 70, 'NewGlucose'] = 'Low'
    data.loc[(data['Glucose'] > 70) & (data['Glucose'] <= 99), 'NewGlucose'] = 'Normal'
    data.loc[(data['Glucose'] > 99) & (data['Glucose'] <= 126), 'NewGlucose'] = 'Overweight'
    
    data['newInsulin'] = data['Insulin'].apply(lambda x: 'Normal' if 16 <= x <= 166 else 'Abnormal')
    
    data['NewBMI'] = 'Very Obese'
    data.loc[data['BMI'] < 18.5, 'NewBMI'] = 'Underweight'
    data.loc[(data['BMI'] >= 18.5) & (data['BMI'] <= 24.9), 'NewBMI'] = 'Normal weight'
    data.loc[(data['BMI'] > 24.9) & (data['BMI'] <= 29.9), 'NewBMI'] = 'Overweight'
    data.loc[(data['BMI'] > 29.9) & (data['BMI'] <= 34.9), 'NewBMI'] = 'Obesity 1'
    data.loc[(data['BMI'] > 34.9) & (data['BMI'] <= 39.9), 'NewBMI'] = 'Obesity 2'
    data.loc[data['BMI'] > 39.9, 'NewBMI'] = 'Obesity 3'
    
    return data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictions', methods=['POST'])
def do_predictions():
    try:
        test_data = request.json
        X_test = pd.DataFrame([test_data])
        
        # Transform features
        X_test = transform_features(X_test)
        
        # Print transformed features
        print("Transformed Features:")
        print(X_test)
        
        # Perform one-hot encoding for categorical features created by transform_features
        X_test = pd.get_dummies(X_test, columns=['NewGlucose', 'newInsulin', 'NewBMI'])
        
        # Ensure all expected columns are present
        for col in expected_features:
            if col not in X_test.columns:
                X_test[col] = 0
        
        # Reorder columns to match the training data
        X_test = X_test[expected_features]
        
        # Print final DataFrame to check columns
        print("Final DataFrame for Prediction:")
        print(X_test)
        
        predictions = model.predict(X_test)
        predictions = predictions.tolist()
        print(predictions)
        return jsonify({"predicted": predictions[0]})
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
