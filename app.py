from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
from pydantic import BaseModel
import joblib
import time
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# Load model once when API starts
model = joblib.load("models/taxi_model.pkl")

# Metrics
prediction_requests = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction latency"
)

# Input schema
class TaxiInput(BaseModel):
    trip_distance: float
    passenger_count: int

@app.get("/")
def home():
    return {"message": "API Running"}

@app.post("/predict")
def predict(data: TaxiInput):

    prediction_requests.inc()

    start = time.time()

    features = [[
        data.trip_distance,
        data.passenger_count
    ]]

    prediction = model.predict(features)

    prediction_latency.observe(time.time() - start)

    return {
        "predicted_fare": float(prediction[0])
    }

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )

Instrumentator().instrument(app).expose(app)  # ← this creates /metrics route