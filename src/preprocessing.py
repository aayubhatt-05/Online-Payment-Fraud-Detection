import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    # Load the dataset
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: '{file_path}' not found. Please check the file path.")
    
    # Check for empty dataset
    if data.empty:
        raise ValueError("Error: The dataset is empty.")
    
    # Handle missing values
    if data.isnull().sum().sum() > 0:
        data = data.dropna()  # Drop rows with missing values (can be replaced with imputation if needed)
    
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Outlier treatment using the IQR method
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    
    # Encoding categorical features
    if 'type' in data.columns:
        type_encoded = pd.get_dummies(data['type'], drop_first=True)
        data = pd.concat([data, type_encoded], axis=1)
    else:
        raise KeyError("Error: 'type' column is missing.")
    
    # Remove irrelevant columns (except 'type' for visualization)
    irrelevant_columns = ['nameOrig', 'nameDest']
    data = data.drop(columns=irrelevant_columns, errors='ignore')
    
    # Feature scaling
    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    # Feature and target separation
    try:
        X = data.drop(['isFraud', 'type'], axis=1)  # Keep 'type' in the dataset but exclude it from features
        y = data['isFraud']
    except KeyError as e:
        raise KeyError(f"Error in feature/target separation: {e}")
    
    return data, X, y