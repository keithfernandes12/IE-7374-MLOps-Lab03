import os
from pathlib import Path

# Paths are built from this file's location, so scripts work from any folder.
# MODEL_DIR can be overridden with an environment variable (e.g. in Docker).
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT_DIR / "model"))
MODEL_FILE = "housing_model.pkl"
METADATA_FILE = "metadata.json"

MODEL_VERSION = "1.0.0"
RANDOM_STATE = 12

# Column order of sklearn's fetch_california_housing; the model expects this order
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"]

# The target is the median house value in units of $100,000, capped at 5.00001
TARGET_UNIT_USD = 100_000
TARGET_CAP = 5.0
