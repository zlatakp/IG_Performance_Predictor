from pathlib import Path
import pandas as pd
from traceback import print_exc

DATA_PATH_INTERIM = Path(Path(__file__).resolve().parents[2]/'data'/'interim')
DATA_PATH_PROCESSED = Path(Path(__file__).resolve().parents[2]/'data'/'processed')

FEATURES = ['Desc length', 'Duration (sec)', 'Post type', 'Day of Week', 'Hour'] #'Permalink'
TARGETS = ['Views', 'Likes', 'Shares', 'Comments', 'Saves', 'Reach', 'Follows']

def build_features():
    try:
        df = pd.read_parquet(DATA_PATH_INTERIM/"joined.parquet")
        df["Day of Week"] = df['Publish time'].dt.day_of_week
        df["Hour"] = df['Publish time'].dt.hour
        df['Desc length'] = df['Description'].str.split().str.len()
        DATA_PATH_PROCESSED.mkdir(parents=True, exist_ok=True)
        df_processed = df[FEATURES+TARGETS]
        df_processed = df_processed.fillna({
            t: 0 for t in TARGETS
        })
        df_processed.to_parquet(DATA_PATH_PROCESSED/"processed.parquet", index=False)
        df_processed.to_csv(DATA_PATH_PROCESSED/"processed.csv", index=False)
    except Exception as e:
        print_exc(e)
if __name__ == '__main__':
    build_features()