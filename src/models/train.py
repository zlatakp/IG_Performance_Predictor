import json, joblib, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from src.features.transformers import preprocess

FEATURES = ['Desc length', 'Duration (sec)', 'Post type', 'Day of Week', 'Hour'] #'Permalink'
TARGETS = ['Views', 'Likes', 'Shares', 'Comments', 'Saves', 'Reach', 'Follows']


DATA_PATH_PROCESSED = Path(Path(__file__).resolve().parents[2]/'data'/'processed')
ART_PATH = Path("artifacts")
MET_PATH = Path("metrics")

TRAINING_SPLIT = 0.8




def splitDf(df: pd.DataFrame) -> list[pd.DataFrame, pd.DataFrame]:
    nrecords = int(len(df)*TRAINING_SPLIT)
    tr, te = df.iloc[:nrecords], df.iloc[nrecords:]
    return tr,te

def train():

    df = pd.read_parquet(DATA_PATH_PROCESSED/"processed.parquet")
    tr, te = splitDf(df)

    xtr, ytr = tr[FEATURES], tr[TARGETS]
    xte, yte = te[FEATURES], te[TARGETS]

    pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", Ridge(alpha=1.0, fit_intercept=True)),
    ])

    pipe.fit(xtr, ytr)

    predicted_data = pd.DataFrame(pipe.predict(xte), columns=TARGETS, index=yte.index)

    rmse = {
        t: float(root_mean_squared_error(yte[t], predicted_data[t])) for t in TARGETS
    }


    pipe_rfg = Pipeline(steps=[
        ("prep", preprocess),
        ("model", RandomForestRegressor(
        n_estimators=100, max_depth=3, n_jobs=-1, random_state=42
    )),
    ])



    pipe_rfg.fit(xtr, ytr)

    predicted_data_rfg = pd.DataFrame(pipe_rfg.predict(xte), columns=TARGETS, index=yte.index)

    rmse_rfg = {
        t: float(root_mean_squared_error(yte[t], predicted_data_rfg[t])) for t in TARGETS
    }


    ART_PATH.mkdir(parents = True, exist_ok=True)
    MET_PATH.mkdir(parents = True, exist_ok=True)

    joblib.dump(pipe, ART_PATH/"model_pipeline.pkl")
    (MET_PATH/"eval.json").write_text(json.dumps({"rmse": rmse}, indent=2))


if __name__ == '__main__':
    train()