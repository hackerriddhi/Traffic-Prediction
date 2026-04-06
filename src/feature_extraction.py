import numpy as np

def create_advanced_features(df):

    df = df.copy()

    if 'traffic' not in df.columns:
        raise ValueError("traffic column missing")

    # Lag feature
    df['lag_1'] = df['traffic'].shift(1)

    # Rolling mean
    df['rolling_mean_3'] = df['traffic'].rolling(window=3).mean()

    # Fill missing
    df = df.bfill()

    return df


def add_external_features(df):

    df = df.copy()

    # Better simulated weather (based on month)
    if 'Month' in df.columns:
        df['weather'] = df['Month'].apply(lambda x: 1 if x in [6,7,8] else 0)
    else:
        df['weather'] = np.random.choice([0,1], size=len(df))

    # Holiday = weekend
    df['is_holiday'] = df['Is_Weekend'] if 'Is_Weekend' in df.columns else 0

    return df