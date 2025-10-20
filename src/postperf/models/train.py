import json, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from features.transformers import preprocess
from features.columns import FEATURES, TARGETS
import pickle

DATA_PATH_PROCESSED = Path(Path(__file__).resolve().parents[3]/'data'/'processed')
ART_PATH = Path("artifacts")
MET_PATH = Path("metrics")

TRAINING_SPLIT = 0.8

def train():
    df = pd.read_parquet(DATA_PATH_PROCESSED/"processed.parquet")
    xtr, xte, ytr, yte = train_test_split(df[FEATURES], df[TARGETS], test_size = 0.2)


    pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", Ridge(alpha=1.0, fit_intercept=True)),
    ])

    pipe.fit(xtr, ytr)

    predicted_data = pd.DataFrame(pipe.predict(xte), columns=TARGETS, index=yte.index)

    rmse = {
        t: float(root_mean_squared_error(yte[t], predicted_data[t])) for t in TARGETS
    }


    # pipe_rfg = Pipeline(steps=[
    #     ("prep", preprocess),
    #     ("model", RandomForestRegressor(
    #     n_estimators=100, max_depth=3, n_jobs=-1, random_state=42
    # )),
    # ])



    # pipe_rfg.fit(xtr, ytr)

    # predicted_data_rfg = pd.DataFrame(pipe_rfg.predict(xte), columns=TARGETS, index=yte.index)

    # rmse_rfg = {
    #     t: float(root_mean_squared_error(yte[t], predicted_data_rfg[t])) for t in TARGETS
    # }


    ART_PATH.mkdir(parents = True, exist_ok=True)
    MET_PATH.mkdir(parents = True, exist_ok=True)

    with open(ART_PATH/"model_pipeline.pkl", "wb") as f:
        pickle.dump(pipe, f)


    with open(MET_PATH/"eval.json", "w") as f:
        f.write(json.dumps({"rmse": rmse}, indent=2))


if __name__ == '__main__':
    train()