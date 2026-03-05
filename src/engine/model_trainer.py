import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_isolation_forest(df_feat, contamination=0.05):
    """
    Train an Isolation Forest model with specified features.
    """
    # 1. FEATURE SELECTION (exclude non-numeric and ground truth)
    features = [
        'price', 'volume', 'hour', 'day_of_week', 'is_after_hours', 
        'price_dev_rolling', 'volume_dev_rolling', 'user_vol_deviation', 
        'price_vol_ratio'
    ]
    
    X = df_feat[features]
    y_true = df_feat['is_anomaly']
    
    # 2. MODEL DEFINITION
    model = IsolationForest(
        n_estimators=200, 
        contamination=contamination, 
        random_state=42,
        max_samples='auto'
    )
    
    # 3. TRAINING
    print(f"Training Isolation Forest with contamination: {contamination}...")
    model.fit(X)
    
    # 4. PREDICTION (-1 for anomaly, 1 for normal)
    # Convert to 1 for anomaly, 0 for normal
    preds_raw = model.predict(X)
    y_pred = np.where(preds_raw == -1, 1, 0)
    
    # Anomaly scores (lower means more anomalous)
    # Re-normalize to 0-1 range for better visualization
    # Decision function returns distance to the frontier
    scores = model.decision_function(X) # Higher score is more normal
    # Smaller values = more anomalous. Let's make it 1 = anomaly, 0 = normal
    # We want to use the scores for alert prioritization!
    df_feat['anomaly_score_distance'] = scores
    # Rescale to 0-1 for easier usage
    df_feat['anomaly_priority'] = 1 - (scores - scores.min()) / (scores.max() - scores.min())
    
    # 5. EVALUATION
    results = classification_report(y_true, y_pred, output_dict=True)
    print("\n--- Model Evaluation Results ---")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    # 6. SAVE MODEL
    joblib.dump(model, "models/anomaly_model.pkl")
    print("\nModel saved to models/anomaly_model.pkl")
    
    return model, df_feat, results

def plot_results(df_eval):
    """
    Plot results for visual inspection.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_eval, 
        x='price', 
        y='volume', 
        hue='anomaly_priority', 
        style='is_anomaly',
        palette='magma',
        alpha=0.7
    )
    plt.title("Detected Anomalies (Isolation Forest Priority)")
    plt.savefig("data/anomaly_plot.png")
    print("Plot saved to data/anomaly_plot.png")

if __name__ == "__main__":
    df = pd.read_csv("data/trading_features_v1.csv")
    
    # Set contamination to roughly the expected fraction of anomalies
    model, df_eval, results = train_isolation_forest(df, contamination=0.1)
    
    # Check prioritization
    print("\nPriority distribution:")
    print(df_eval.groupby('anomaly_type')['anomaly_priority'].mean())
    
    # Plot
    plot_results(df_eval)
    
    # Save results
    df_eval.to_csv("data/trading_evaluation_v1.csv", index=False)
