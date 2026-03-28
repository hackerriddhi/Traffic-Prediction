import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    
    # Loads the raw traffic dataset and performs basic cleaning.
    
    print("Loading data from:", filepath)
    df = pd.read_csv(filepath)
    
    # Drop the ID column as it doesn't provide predictive value
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        
    # Convert DateTime column to actual Pandas Datetime objects
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Sort by time to ensure time-series integrity
    df = df.sort_values(by=['Junction', 'DateTime']).reset_index(drop=True)
    
    # Check for missing values
    if df.isnull().sum().any():
        print("Warning: Missing values detected. Forward filling...")
        df = df.ffill()
        
    print("Data loaded and cleaned successfully!")
    return df

def extract_time_features(df):
    """
    Extracts time-based features from the DateTime column.
    This is crucial for Pattern Recognition in time series.
    """
    print("Extracting time features...")
    df_features = df.copy()
    
    # Basic Time Features
    df_features['Hour'] = df_features['DateTime'].dt.hour
    df_features['Day'] = df_features['DateTime'].dt.day
    df_features['DayOfWeek'] = df_features['DateTime'].dt.dayofweek # 0=Monday, 6=Sunday
    df_features['Month'] = df_features['DateTime'].dt.month
    df_features['Year'] = df_features['DateTime'].dt.year
    
    #  Custom Features
    # 1. Is it the weekend? (Saturday=5, Sunday=6)
    df_features['Is_Weekend'] = df_features['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 2. Is it a peak hour? (Assuming 8-10 AM and 5-7 PM (17-19) are peak traffic)
    df_features['Is_Peak_Hour'] = df_features['Hour'].apply(
        lambda x: 1 if (8 <= x <= 10) or (17 <= x <= 19) else 0
    )
    
    print("Time features extracted!")
    return df_features

if __name__ == "__main__":
    # This block runs if you execute `python src/data_preprocessing.py`
    # 1. Load raw data
    raw_path = "../data/raw/traffic.csv"
    processed_path = "../data/processed/cleaned_traffic.csv"
    
    df_cleaned = load_and_clean_data(raw_path)
    
    # 2. Extract features
    df_final = extract_time_features(df_cleaned)
    
    # 3. Save to processed folder 
    df_final.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}.")