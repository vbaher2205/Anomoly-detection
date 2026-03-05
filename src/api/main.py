from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
from src.alerts.alert_manager import AlertManager

app = FastAPI(title="Market Anomaly Detection API")
alert_manager = AlertManager()

# Data Model
class Alert(BaseModel):
    alert_id: str
    timestamp: str
    user_id: str
    stock_id: str
    priority_score: float
    severity: str
    details: str
    status: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Market Anomaly Detection API"}

@app.get("/alerts", response_model=List[Alert])
def get_alerts(threshold: float = 0.6):
    """
    Fetch all prioritize alerts above a certain threshold.
    """
    alerts = alert_manager.generate_alerts(threshold=threshold)
    return alerts

@app.post("/alerts/refresh")
def refresh_alerts():
    """
    Reload alerts from the latest evaluation data.
    """
    try:
        alert_manager.generate_alerts()
        alert_manager.save_alerts()
        return {"status": "success", "count": len(alert_manager.alerts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/alerts/{alert_id}/status")
def update_alert_status(alert_id: str, status: str):
    """
    Update the investigation status of an alert (e.g., CLOSED, UNDER_INVESTIGATION).
    """
    # In a real app, this would update a database
    return {"alert_id": alert_id, "new_status": status, "message": "Status updated successfully (mock)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
