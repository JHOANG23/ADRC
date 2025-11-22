import os
import torch
import opensmile
import numpy as np
import pandas as pd
from src.utils import split_wav

def valid_row(row, append_cov, covariates):
    if not row.file_name.endswith(".wav") or row.synd2 == -1.0 or np.isnan(row.synd2):
        return False
    if append_cov:
        for cov in covariates:
            val = getattr(row, cov, None)
            if pd.isna(val):
                return False
    return True

# extract features for all wav files
def egemaps_extract(
        df: pd.DataFrame, 
        input_dir: str, 
        mode: str, 
        N: int, 
        append_cov: bool, 
        covariates: list = None 
    ): 
    # Define what features we want extracted:
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors
    )

    csv_rows = []
    for row in df.itertuples(index=False):
        if not valid_row(row, append_cov, covariates):
            continue 
        print(f"Extracting LLD egemaps feature from: {row.file_name}")            
        wav_file = os.path.join(input_dir, row.language, row.file_name)
        segments, sr = split_wav(wav_file, mode, N)         

        for segment in segments:
            y = smile.process_signal(segment, sr)
            mean_features = y.mean().add_prefix("mean_")
            std_features = y.std().add_prefix("std_")

            feature_dict = {"file_name":row.file_name, "language":row.language, "label": row.synd2,}
            feature_dict.update(mean_features.to_dict())
            feature_dict.update(std_features.to_dict())
            
            if append_cov:
                for cov in covariates:
                    feature_dict[cov] = getattr(row, cov)

            csv_rows.append(feature_dict)

    return csv_rows

