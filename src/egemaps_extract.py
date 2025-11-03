import os
import torch
import audiofile
import opensmile
import numpy as np
import pandas as pd
from config import TRIMMED_AUDIO_PATH

def _egemaps_extract(wav_file, smile): #extract features for a singular wav file
    signal, sampling_rate = audiofile.read(
        wav_file, 
        always_2d=True
    )
    feature_row = smile.process_signal(signal, sampling_rate)
    return feature_row

def egemaps_extract(df, input_dir, isSplit=False): # extract features for all wav files
    # Define waht features we want extracted:
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    X = []
    Y = []
    for row in df.itertuples(index=False):
        print(f"Extracting egemaps feature from: {row.file_name}")
        if not row.file_name.endswith(".wav") or row.synd2 == -1.0 or np.isnan(row.synd2):
            continue
        if isSplit == True: # Some papers segment the audio file before extracting features for all segments
            pass            # ill touch this later when needed
        
        wav_file = os.path.join(input_dir, row.language, row.file_name)
        feature = _egemaps_extract(wav_file, smile)
        X.append(feature)
        Y.append(row.synd2)

    X = np.vstack(X)
    print(f"X shape: {X.shape}")
    X_tensor = torch.from_numpy(X).float().cpu()
    Y = np.array(Y)
    print(f"Y shape: {Y.shape}")
    Y_tensor = torch.from_numpy(Y).long().cpu()

    return X_tensor, Y_tensor
