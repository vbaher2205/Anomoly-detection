import pandas as pd
import uuid
from datetime import datetime

class AlertManager:
    def __init__(self, data_path="data/trading_evaluation_v1.csv"):
        self.data_path = data_path
        self.alerts = []

    def generate_alerts(self, threshold=0.6):
        """
        Processes evaluation data and generates alerts for records exceeding the priority threshold.
        """
        df = pd.read_csv(self.data_path)
        
        # Filter records exceeding threshold
        anomalies = df[df['anomaly_priority'] >= threshold].copy()
        
        alerts = []
        for _, row in anomalies.iterrows():
            alert = {
                "alert_id": str(uuid.uuid4())[:8],
                "timestamp": row['timestamp'],
                "user_id": row['user_id'],
                "stock_id": row['stock_id'],
                "priority_score": round(row['anomaly_priority'], 4),
                "severity": self._assign_severity(row['anomaly_priority']),
                "details": f"Price: {row['price']:.2f}, Volume: {row['volume']:.2f}, Type: {row['anomaly_type']}",
                "status": "OPEN"
            }
            alerts.append(alert)
            
        self.alerts = sorted(alerts, key=lambda x: x['priority_score'], reverse=True)
        return self.alerts

    def _assign_severity(self, score):
        if score >= 0.9:
            return "CRITICAL"
        elif score >= 0.75:
            return "HIGH"
        elif score >= 0.6:
            return "MEDIUM"
        return "LOW"

    def save_alerts(self, output_path="data/active_alerts.csv"):
        if not self.alerts:
            print("No alerts to save.")
            return
        df_alerts = pd.DataFrame(self.alerts)
        df_alerts.to_csv(output_path, index=False)
        print(f"Saved {len(self.alerts)} alerts to {output_path}")

if __name__ == "__main__":
    manager = AlertManager()
    alerts = manager.generate_alerts(threshold=0.6)
    print(f"Generated {len(alerts)} alerts.")
    if alerts:
        print("\nTop 5 Alerts:")
        for a in alerts[:5]:
            print(f"[{a['severity']}] {a['alert_id']} - User: {a['user_id']}, Score: {a['priority_score']}")
    manager.save_alerts()
