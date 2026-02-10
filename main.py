from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI(title="Indoor Localisation API")

model = joblib.load("./models/linear_model1.pkl")


# ----------- Nested models -----------
class MagData(BaseModel):
    x: float
    y: float
    z: float
    ts: int | None = None


class GpsData(BaseModel):
    lat: float
    lon: float
    ts: int | None = None


class PredictRequest(BaseModel):
    room: str | None = None
    mag: MagData
    gps: GpsData


@app.post("/predict")
def predict(req: PredictRequest):
    print("REQUEST RECEIVED:", req)

    X = [[
        req.gps.lat,
        req.gps.lon,
        req.mag.x,
        req.mag.y,
        req.mag.z,
        (req.mag.x**2 + req.mag.y**2 + req.mag.z**2) ** 0.5  # magnitude
    ]]

    y = model.predict(X)

    return {
        "ok": True,
        "x": float(y[0][0]),
        "y": float(y[0][1]),
    }


@app.post("/hello_world")
def hello_world():
    return {"hello": "world"}



#to run
#python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload