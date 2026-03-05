import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_trading_data(num_samples=1000):
    """
    Generates synthetic trading activity data.
    - timestamp
    - user_id
    - stock_id
    - price
    - volume
    - is_anomaly (ground truth)
    """
    np.random.seed(42)
    random.seed(42)
    
    start_time = datetime(2026, 1, 1, 9, 30) # Market open
    data = []
    
    # 1. GENERATE NORMAL DATA
    for i in range(num_samples):
        # Normal distribution centered around 100 for price and 1000 for volume
        ts = start_time + timedelta(minutes=random.randint(0, 1440 * 10))
        user_id = f"USER_{random.randint(1, 100):03d}"
        stock_id = f"STOCK_{random.randint(1, 10):02d}"
        price = max(10, np.random.normal(150, 20))
        volume = max(1, np.random.normal(2000, 500))
        
        data.append({
            'timestamp': ts,
            'user_id': user_id,
            'stock_id': stock_id,
            'price': price,
            'volume': volume,
            'is_anomaly': 0,
            'anomaly_type': 'none'
        })
        
    df = pd.DataFrame(data)
    
    # 2. INJECT POINT ANOMALIES (Extreme price/volume)
    num_points = int(num_samples * 0.02)
    point_indices = np.random.choice(df.index, num_points, replace=False)
    for idx in point_indices:
        df.at[idx, 'price'] *= random.uniform(5, 10)
        df.at[idx, 'volume'] *= random.uniform(10, 20)
        df.at[idx, 'is_anomaly'] = 1
        df.at[idx, 'anomaly_type'] = 'point'
        
    # 3. INJECT CONTEXTUAL ANOMALIES (Unusual time)
    # E.g., High activity during non-trading hours (between 11 PM and 5 AM)
    num_contextual = int(num_samples * 0.02)
    context_indices = np.random.choice(df[df['is_anomaly'] == 0].index, num_contextual, replace=False)
    for idx in context_indices:
        h = random.randint(23, 23) if random.random() > 0.5 else random.randint(0, 4)
        m = random.randint(0, 59)
        new_ts = df.at[idx, 'timestamp'].replace(hour=h, minute=m)
        df.at[idx, 'timestamp'] = new_ts
        df.at[idx, 'price'] *= random.uniform(1.5, 2.5) # Slight elevation
        df.at[idx, 'is_anomaly'] = 1
        df.at[idx, 'anomaly_type'] = 'contextual'
        
    # 4. INJECT COLLECTIVE ANOMALIES (Layering/Spoofing mock)
    # A single user makes 10 trades in 1 minute with small increments
    num_collective_groups = 3
    for _ in range(num_collective_groups):
        user_id = "USER_ROGUE"
        stock_id = "STOCK_TARGET"
        collective_start = start_time + timedelta(minutes=random.randint(0, 1440 * 10))
        for i in range(10):
            ts = collective_start + timedelta(seconds=i*5)
            df = pd.concat([df, pd.DataFrame([{
                'timestamp': ts,
                'user_id': user_id,
                'stock_id': stock_id,
                'price': 150 + (i * 0.5),
                'volume': 1000,
                'is_anomaly': 1,
                'anomaly_type': 'collective'
            }])], ignore_index=True)
            
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_trading_data()
    df.to_csv("data/trading_activity_v1.csv", index=False)
    print(f"Generated {len(df)} samples in data/trading_activity_v1.csv")
    print(df['anomaly_type'].value_counts())
