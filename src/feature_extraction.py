import pandas as pd

def create_advanced_features(df):

    # Lag feature
    df['lag_1'] = df['traffic'].shift(1)

    # Rolling mean
    df['rolling_mean_3'] = df['traffic'].rolling(window=3).mean()

    # ✅ FIX: correct column name
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['hour'] = df['DateTime'].dt.hour
        df['day_of_week'] = df['DateTime'].dt.dayofweek

    # Fill missing values
    df = df.bfill()

    return df