import pandas as pd
from pathlib import Path
from logging import basicConfig
import logging
from datetime import datetime
from traceback import print_exc
import os
from typing import List
USECOLS = ['Post ID', 'Description', 'Duration (sec)', 'Publish time', 'Permalink', 'Post type', 'Views', 'Likes', 'Shares', 'Comments', 'Saves', 'Reach', 'Follows']
DATA_PATH_RAW = Path(Path(__file__).resolve().parents[2]/'data'/'raw')
DATA_PATH_INTERIM = Path(Path(__file__).resolve().parents[2]/'data'/'interim')
TIMESTAMP = datetime.now().strftime("%d-%m-%Y %H_%M_%S")
basicConfig(filename=Path(f"logging/{TIMESTAMP}.log"), level="INFO", format="%(asctime)s: %(message)s")
log = logging.getLogger(__name__)


def build() -> None:
    df_l, df_r, field = load_data()
    df_merge = df_l.merge(df_r, how="left", on=field)
    DATA_PATH_INTERIM.mkdir(exist_ok=True)
    df_merge.drop(columns=[field])
    df_merge.to_parquet(DATA_PATH_INTERIM/"joined.parquet", index = False)
    log.info("merge and saved")

def load_data()-> list[pd.DataFrame, pd.DataFrame, str]:
    try:
        merge_field = "Date"
        df_content = pd.read_csv(DATA_PATH_RAW/"Content.csv", usecols=USECOLS, parse_dates=['Publish time'])
        df_content[merge_field] = pd.to_datetime(df_content["Publish time"]).dt.floor("D")
        df_follows = pd.read_csv(DATA_PATH_RAW/"Follows.csv", parse_dates=[merge_field])
        return df_content, df_follows, merge_field
    except Exception as e:
        print_exc()
        log.info(str(e))



if __name__ == '__main__':
    build()