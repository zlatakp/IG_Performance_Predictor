from pathlib import Path
import joblib, json, pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform
import sys
sys.path.insert(0, '../features')
from transformers import preprocess

DATA_PATH_PROCESSED = Path(Path(__file__).resolve().parents[2]/'data'/'processed')
TARGETS = ['Views', 'Likes', 'Shares', 'Comments', 'Saves', 'Reach', 'Follows']
ART_PATH = Path("artifacts")
MET_PATH = Path("metrics")

def tune():
    df = pd.read_parquet(DATA_PATH_PROCESSED/"processed.parquet")
    x, y = df.drop(columns=TARGETS), df[TARGETS]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", Ridge(alpha=1.0)),
    ])

    search = RandomizedSearchCV(
        estimator = pipe,
        param_distributions={
            "model__alpha": loguniform(1e-4, 1e3),
            "model__fit_intercept": [True, False]
        },
        scoring = "neg_root_mean_squared_error",
        n_iter=40,
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    search.fit(x, y)

    ART_PATH.mkdir(parents=True, exist_ok=True)
    MET_PATH.mkdir(parents=True, exist_ok=True)

    joblib.dump(search.best_estimator_, ART_PATH/"model_pipeline_best.pkl")
    (MET_PATH/"cv_results.json").write_text(json.dumps({
        "best_params": search.best_params_,
        "best_rmse": -float(search.best_score_)
    }, indent = 2))

if __name__ == '__main__':
    tune()