import pandas as pd

def create_advanced_features(df):

    # Lag feature
    df['lag_1'] = df['traffic'].shift(1)

    # Rolling mean
    df['rolling_mean_3'] = df['traffic'].rolling(window=3).mean()

    
    # Fill missing values
    df = df.bfill()

    return df