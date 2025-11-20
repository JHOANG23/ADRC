import os
import torch
import opensmile
import numpy as np
import pandas as pd
from src.utils import split_wav

def egemaps_extract(df, input_dir, mode='equal', N=1):  # extract features for all wav files
    # Define what features we want extracted:
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors
    )
    embeddings = {'en': [], 'es': []}
    labels = {'en': [], 'es': []}
    languages = ['en', 'es']
    corresponding_files = {'en': [], 'es': []}
    
    for lang in languages:
        for row in df[df['language'] == lang].itertuples(index=False):
            if not row.file_name.endswith(".wav") or row.synd2 == -1.0 or np.isnan(row.synd2):
                continue
            if pd.isna(row.age_at_recording) or pd.isna(row.Education) or pd.isna(row.Gender):
                continue
            
            print(f"Extracting egemaps feature from: {row.file_name}")            
            wav_file = os.path.join(input_dir, row.language, row.file_name)
            segments, sr = split_wav(wav_file, mode, N)         
            
            file_segment_features = []
            covariates = np.array([row.age_at_recording, row.Education, row.Gender], dtype=float)
            for segment in segments:
                _, _, y = smile.process(segment, sr)
                feature = y[0, :]
                # --- SAFEGUARD: skip if any NaN (too-short segment) ---
                if np.isnan(feature).any():
                    continue
                feature_cov = np.concatenate((feature, covariates))  
                file_segment_features.append(torch.from_numpy(feature_cov).float())

            # Skip files with no valid segments
            if not file_segment_features:
                continue

            sequence_tensor = torch.stack(file_segment_features, dim=0)
            embeddings[lang].append(sequence_tensor)
            labels[lang].append(row.synd2) 
            corresponding_files[lang].append(row.file_name)

    X_en_tensor = torch.stack(embeddings['en'], dim=0)
    Y_en_tensor = torch.tensor(labels['en'], dtype=torch.long)
    X_es_tensor = torch.stack(embeddings['es'], dim=0)
    Y_es_tensor = torch.tensor(labels['es'], dtype=torch.long)

    return X_en_tensor, Y_en_tensor, X_es_tensor, Y_es_tensor, corresponding_files

def test_import():
    print('TESTING!')
