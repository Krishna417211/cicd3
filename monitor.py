

"""
Simple monitoring: check the model loads and predicts,
and check whether Prometheus and Grafana are reachable.
"""
import os
import joblib

print("=" * 50)
print("Monitoring")
print("=" * 50)

# 1. Model health
if os.path.exists("models/taxi_model.pkl"):
    model = joblib.load("models/taxi_model.pkl")
    pred = model.predict([[1, 2.5, 10.0, 2.0]])
    print(f"[OK]  Model loads and predicts: {pred[0]:.2f}")
else:
    print("[DOWN] Model file not found")

# 2. Monitoring services (only up when docker-compose is running)
try:
    import requests
    for name, url in {
        "Prometheus": "http://localhost:9090/-/healthy",
        "Grafana": "http://localhost:3000/api/health",
    }.items():
        try:
            r = requests.get(url, timeout=5)
            print(f"[OK]  {name} reachable" if r.status_code == 200
                  else f"[WARN] {name} status {r.status_code}")
        except Exception:
            print(f"[DOWN] {name} not reachable (start docker-compose to check)")
except ImportError:
    print("requests not installed - skipping URL checks")

print("=" * 50)
print("Monitoring done")
print("=" * 50)