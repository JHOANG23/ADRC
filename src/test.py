import os
import torch
import pandas as pd
from src.utils import format_df
from src.config import EXCEL_FULL_ADRC_PATH, TRIMMED_AUDIO_PATH, FEATURE_PATH
from src.feature_extraction.egemaps_extract import egemaps_extract

df = format_df(EXCEL_FULL_ADRC_PATH)
mode = 'equal'
N = 1
csv_out = f"{FEATURE_PATH}/egemapsv02_{mode}_n{N}/features.csv"
csv_rows = egemaps_extract(
    df=df, 
    input_dir=TRIMMED_AUDIO_PATH, 
    mode=mode, 
    N=N,
    append_cov=False,
)

df_features = pd.DataFrame(csv_rows)
df_features.to_csv(csv_out, index=False)






