import pandas as pd
import numpy as np

def extract_features(df):
    """
    Extract features for anomaly detection.
    - hour_of_day, day_of_week
    - rolling price/volume means (1 min, 5 min)
    - price/volume deviations from rolling means
    - price/volume momentum
    """
    df = df.copy()
    
    # 1. TIME FEATURES
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Is it after hours? (e.g. 10 PM - 8 AM)
    df['is_after_hours'] = ((df['hour'] < 9) | (df['hour'] >= 22)).astype(int)
    
    # 2. ROLLING FEATURES (per stock)
    # Using window-based features requires sorting and potentially some delay in real-time, 
    # but here we use the historical window.
    df = df.sort_values(['stock_id', 'timestamp'])
    
    # Rolling 10 samples (simulated for immediate proximity)
    df['rolling_price_mean'] = df.groupby('stock_id')['price'].shift(1).rolling(window=10).mean()
    df['rolling_volume_mean'] = df.groupby('stock_id')['volume'].shift(1).rolling(window=10).mean()
    
    df['price_dev_rolling'] = df['price'] / df['rolling_price_mean']
    df['volume_dev_rolling'] = df['volume'] / df['rolling_volume_mean']
    
    # 3. USER BEHAVIOR (per user)
    # Average trade size for the user
    user_stats = df.groupby('user_id')['volume'].transform('mean')
    df['user_vol_deviation'] = df['volume'] / user_stats
    
    # 4. PRICE-VOLUME COUPLING
    df['price_vol_ratio'] = df['price'] * df['volume']
    
    # Clean up NaNs from rolling windows
    df = df.fillna(0)
    
    return df

if __name__ == "__main__":
    df_raw = pd.read_csv("data/trading_activity_v1.csv", parse_dates=['timestamp'])
    print("Preprocessing data...")
    df_feat = extract_features(df_raw)
    df_feat.to_csv("data/trading_features_v1.csv", index=False)
    print("Features extracted and saved to data/trading_features_v1.csv")
    print(df_feat.columns)
