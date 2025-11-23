import os
import torch
import opensmile
import numpy as np
import pandas as pd
from src.utils import split_wav
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import MidTermFeatures as m_aF

def valid_row(row, append_cov, covariates):
    if not row.file_name.endswith(".wav") or row.synd2 == -1.0 or np.isnan(row.synd2):
        return False
    if append_cov:
        for cov in covariates:
            val = getattr(row, cov, None)
            if pd.isna(val):
                return False
    return True

def pyaudio_extract(df: pd.DataFrame, input_dir: str, mode: str, N: int):
    csv_rows = []

    for row in df.itertuples(index=False):
        if not row.file_name.endswith(".wav") or row.synd2 == -1.0 or np.isnan(row.synd2):
            continue

        print(f"Extracting pyAudio features from: {row.file_name}")
        audio_path = os.path.join(input_dir, row.language, row.file_name)
        segments, sr = split_wav(audio_path, mode, N)
        for segment in segments:
            fs = sr
            x = segment

            # mid_window=1s, mid_step=1s, short_window=50ms, short_step=50ms
            mt, st, mt_n = m_aF.mid_feature_extraction(
                x,
                fs,
                1 * fs,
                1 * fs,
                0.05 * fs,
                0.05 * fs
            )

            # Average across mid-term frames to get one feature vector
            feature_vector = np.mean(mt, axis=1)
            feature_dict = {
                "file_name": row.file_name,
                "language": row.language,
                "label": row.synd2,
            }

            for feat_name, value in zip(mt_n, feature_vector):
                feature_dict[f"{feat_name}"] = float(value)

            csv_rows.append(feature_dict)
    return csv_rows
