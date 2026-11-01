from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from config import MODEL_DIR, MODEL_VERSION, TARGET_UNIT_USD
from predict import input_warnings, load_model, predict_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup instead of on every request."""
    try:
        app.state.model, app.state.metadata = load_model(MODEL_DIR)
    except FileNotFoundError:
        # Start anyway so GET / can report the problem; prediction routes return 503
        app.state.model, app.state.metadata = None, None
    yield


app = FastAPI(
    title="California Housing Price API",
    description="Predicts the median house value of a California census block group "
                "from 1990 census features.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


class HousingData(BaseModel):
    """
    Pydantic BaseModel describing one California census block group.

    Hard limits reject impossible values with a 422. Values that are possible but rare
    in the training data are accepted and flagged in the response warnings instead.
    """
    med_inc: float = Field(gt=0, le=20, description="Median household income, in tens of thousands of USD")
    house_age: float = Field(ge=0, le=100, description="Median house age, in years")
    ave_rooms: float = Field(gt=0, le=200, description="Average rooms per household")
    ave_bedrms: float = Field(gt=0, le=50, description="Average bedrooms per household")
    population: float = Field(ge=1, le=100_000, description="Block group population")
    ave_occup: float = Field(gt=0, le=2_000, description="Average household members")
    latitude: float = Field(ge=32.5, le=42.0, description="Latitude, within California")
    longitude: float = Field(ge=-124.5, le=-114.0, description="Longitude, within California")

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "med_inc": 8.3252, "house_age": 41.0, "ave_rooms": 6.98, "ave_bedrms": 1.02,
        "population": 322.0, "ave_occup": 2.56, "latitude": 37.88, "longitude": -122.23,
    }]})

    @model_validator(mode="after")
    def bedrooms_within_rooms(self):
        if self.ave_bedrms > self.ave_rooms:
            raise ValueError("ave_bedrms cannot be greater than ave_rooms")
        return self

    def to_row(self):
        """Return the features in the order the model was trained on (config.FEATURES)."""
        return [self.med_inc, self.house_age, self.ave_rooms, self.ave_bedrms,
                self.population, self.ave_occup, self.latitude, self.longitude]


class BatchRequest(BaseModel):
    samples: list[HousingData] = Field(min_length=1, max_length=100)


class Prediction(BaseModel):
    median_house_value: float = Field(description="Predicted median house value, in USD")
    warnings: list[str] = []


class PredictionResponse(Prediction):
    model_version: str


class BatchResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    model_version: str
    model_name: str
    trained_at: str
    sklearn_version: str
    n_train: int
    n_test: int
    features: list[str]
    cv_rmse: dict[str, dict[str, float]]
    test_metrics: dict[str, float]
    feature_ranges: dict[str, dict[str, float]]


def get_model(request: Request):
    """Dependency that returns the loaded model, or a 503 if it failed to load."""
    if request.app.state.model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Model is not loaded. Run src/train.py and restart the API.")
    return request.app.state.model


def score(model, metadata, samples):
    """Predict a list of HousingData and attach warnings to each prediction."""
    rows = [s.to_row() for s in samples]
    preds = predict_data(model, rows)
    return [Prediction(median_house_value=round(float(p) * TARGET_UNIT_USD, 2),
                       warnings=input_warnings(row, float(p), metadata["feature_ranges"]))
            for row, p in zip(rows, preds)]


# Routes use plain `def`: model.predict is blocking CPU work, so FastAPI runs these in a
# thread pool instead of blocking the event loop as `async def` would.

@app.get("/", status_code=status.HTTP_200_OK, tags=["health"])
def health_ping(request: Request):
    return {"status": "healthy", "model_loaded": request.app.state.model is not None}


@app.get("/model/info", response_model=ModelInfo, tags=["model"])
def model_info(request: Request, model=Depends(get_model)):
    """Version, training date and evaluation metrics of the served model."""
    return request.app.state.metadata


@app.post("/predict", response_model=PredictionResponse, tags=["predict"])
def predict_price(housing: HousingData, request: Request, model=Depends(get_model)):
    """
    Predict the median house value of one census block group.

    Example response:
        {"median_house_value": 418543.21, "warnings": [], "model_version": "1.0.0"}
    """
    metadata = request.app.state.metadata
    [prediction] = score(model, metadata, [housing])
    return PredictionResponse(**prediction.model_dump(), model_version=metadata["model_version"])


@app.post("/predict/batch", response_model=BatchResponse, tags=["predict"])
def predict_batch(batch: BatchRequest, request: Request, model=Depends(get_model)):
    """Predict up to 100 census block groups in one call, in the order given."""
    metadata = request.app.state.metadata
    return BatchResponse(model_version=metadata["model_version"],
                         predictions=score(model, metadata, batch.samples))
